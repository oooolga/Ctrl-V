from diffusers import UNet2DConditionModel as UNet2DConditionModel_original
import torch
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.embeddings import TextTimeEmbedding

class UNet2DConditionModel(UNet2DConditionModel_original):

    def set_disable_object_condition(self, disable_object_condition: bool):
        self.disable_object_condition = disable_object_condition
    
    def preprocess_object_embdes(self, object_embeds: Optional[torch.Tensor]):
        if object_embeds is None: return object_embeds
        return torch.zeros_like(object_embeds) if self.disable_object_condition else object_embeds

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
        ) -> None:
        if encoder_hid_dim_type == "text_object_proj":
            self.object_u = nn.Parameter(torch.abs(torch.randn(1)*0.1), requires_grad=True)
        else:
            super()._set_encoder_hid_proj(encoder_hid_dim_type, cross_attention_dim, encoder_hid_dim)
    
    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "object":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
            self.object_w = nn.Parameter(torch.abs(torch.randn(1)*0.1), requires_grad=True)
            self.add_embedding_init()
        else:
            super()._set_add_embedding(
                addition_embed_type, addition_embed_type_num_heads, addition_time_embed_dim, flip_sin_to_cos, freq_shift, cross_attention_dim, encoder_hid_dim, projection_class_embeddings_input_dim, time_embed_dim
            )
    
    def add_embedding_init(self):
        for name, param in self.add_embedding.named_parameters():
            try:
                if "weight" in name:
                    torch.nn.init.xavier_normal_(param)
                elif "bias" in name:
                    torch.nn.init.constant_(param, 0.0)
            except:
                torch.nn.init.normal_(param)

    def enable_object_em_gradients(self):
        if self.disable_object_condition:
            return 
        self.object_w.requires_grad_(True)
        if hasattr(self, 'object_u'):
            self.object_u.requires_grad_(True)
        for param in self.add_embedding.parameters():
            param.requires_grad_(True)
    
    def get_object_u_value(self):
        return self.object_u.cpu().detach().numpy()[0]
    
    def get_object_w_value(self):
        return self.object_w.cpu().detach().numpy()[0]
    
    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ):
        if self.config.addition_embed_type == "object":
            object_embs = added_cond_kwargs.get("object_embeds")
            aug_emb = self.object_w*self.add_embedding(object_embs)
            return aug_emb
        return super().get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs)
    
    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.config.encoder_hid_dim_type == "text_object_proj":
            object_embs, negative_object_embs = added_cond_kwargs.get("object_embeds"), added_cond_kwargs.get("negative_object_embeds")
            if negative_object_embs is None:
                encoder_hidden_states = encoder_hidden_states + self.object_u*object_embs
            else:
                encoder_hidden_states = encoder_hidden_states + self.object_u*torch.cat([negative_object_embs, object_embs])
            return encoder_hidden_states
        return super().process_encoder_hidden_states(encoder_hidden_states, added_cond_kwargs)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        object_embs = added_cond_kwargs.get("object_embeds")
        added_cond_kwargs["object_embeds"] = self.preprocess_object_embdes(object_embs)
        return super().forward(
            sample, timestep, encoder_hidden_states, class_labels, timestep_cond, attention_mask, cross_attention_kwargs, added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, down_intrablock_additional_residuals, encoder_attention_mask, return_dict
        )