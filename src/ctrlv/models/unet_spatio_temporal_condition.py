
from diffusers.loaders import PeftAdapterMixin

from diffusers import UNetSpatioTemporalConditionModel as UNetSpatioTemporalConditionModel_orig
from .attention import BBOXFrameAttention
import torch
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput


class UNetSpatioTemporalConditionModel(UNetSpatioTemporalConditionModel_orig, PeftAdapterMixin):
    
    def enable_grad(self, temporal_transformer_block=True, all=False):
        parameters_list = []
        for name, param in self.named_parameters():
            if bool('temporal_transformer_block' in name and temporal_transformer_block) or all:
                parameters_list.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        return parameters_list
    
    def get_parameters_with_grad(self):
        return [param for param in self.parameters() if param.requires_grad]
    
    def encode_bbox_frame(self, frame_latent, encoded_objects):
        return frame_latent.unsqueeze(1).repeat(1, self.config.num_frames, 1, 1, 1)
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residuals: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        is_controlnet = mid_block_additional_residuals is not None and down_block_additional_residuals is not None

        # 1. time
        timesteps = timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples
        
        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )
        if is_controlnet:
            sample = sample + mid_block_additional_residuals

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)

class UNetSpatioTemporalConditionModel_with_bbox_cond(UNetSpatioTemporalConditionModel):
    @register_to_config
    def __init__(
            self, 
            sample_size: Optional[int] = None,
            in_channels: int = 8,
            out_channels: int = 4,
            down_block_types: Tuple[str] = (
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            up_block_types: Tuple[str] = (
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            addition_time_embed_dim: int = 256,
            projection_class_embeddings_input_dim: int = 768,
            layers_per_block: Union[int, Tuple[int]] = 2,
            cross_attention_dim: Union[int, Tuple[int]] = 1024,
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
            num_frames: int = 25,
        ):
        super().__init__(sample_size=sample_size,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         down_block_types=down_block_types,
                         up_block_types=up_block_types,
                         block_out_channels=block_out_channels,
                         addition_time_embed_dim=addition_time_embed_dim,
                         projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                         layers_per_block=layers_per_block,
                         cross_attention_dim=cross_attention_dim,
                         transformer_layers_per_block=transformer_layers_per_block,
                         num_attention_heads=num_attention_heads,
                         num_frames=num_frames)
        self.bbox_frame_attention = BBOXFrameAttention(
            num_frames=self.config.num_frames,
            in_channels=self.config.out_channels,
            out_channels=self.config.out_channels*self.config.num_frames,
            num_layers=8,
            cross_attention_dim=None,
            norm_num_groups=4,
            use_linear_projection=False,
            upcast_attention=False,
            attention_type='default'
        )
    
    def get_attention_rz_weight(self, if_absolute=False):
        return torch.abs(self.bbox_frame_attention.attn.rz_weight.detach()).cpu().item() if if_absolute else self.bbox_frame_attention.attn.rz_weight.detach().cpu().item()
    
    def encode_bbox_frame(self, frame_latent, encoded_objects):
        batch_size, num_frames, num_objects, object_dim = encoded_objects.shape
        _, C, H, W = frame_latent.shape
        encoded_objects = rearrange(encoded_objects, 'b f o d -> b (f o) d')
        frame_latent_copy = frame_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        result = self.bbox_frame_attention(frame_latent, encoded_objects).reshape(batch_size, num_frames, C, H, W)
        # return torch.cat([frame_latent_copy, result], dim=2)
        return result

    def enable_grad_bbox_frame_embedder(self):
        parameters_list = []
        for name, param in self.bbox_frame_attention.named_parameters():
            parameters_list.append(param)
            param.requires_grad = True
        return parameters_list