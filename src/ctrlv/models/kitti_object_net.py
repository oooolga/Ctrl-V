from ctrlv.utils import FourierEmbedder
import torch.nn as nn
import torch
from diffusers import ModelMixin

class KittiObjectNet(ModelMixin):

    FOURIER_FREQS = 32

    def __init__(self, out_dim, num_id_classes=9, num_occluded_classes=5, mid_dim=2048):
        super().__init__()

        # EMBED TYPE                   DIMS (FF=Fourier freqs)
        # -----------------------------------------------------
        # truncated                    FF*2
        # occluded                     FF*2 (embed)
        # alpha                        FF*2
        # bbox                         FF*2*4
        # dimensions                   FF*2*3
        # locations                    FF*2*3
        # rotation_y                   FF*2
        # id_type                      FF*2 (embed)
        # -----------------------------------------------------
        # TOTAL                        FF*30

        self.fourier_embedder = FourierEmbedder(num_freqs=self.FOURIER_FREQS)
        self.id_embedder = nn.Embedding(num_id_classes, 2*self.FOURIER_FREQS)
        self.occluded_embedder = nn.Embedding(num_occluded_classes, 2*self.FOURIER_FREQS)
        self.input_dim = 30*self.FOURIER_FREQS

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, out_dim)
        )

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, x):
        original_shape = x['id_type'].shape
        is_clip = len(original_shape) == 3
        if is_clip:
            for key in ['truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'locations', 'rotation_y', 'id_type']:
                s = x[key].shape
                x[key] = x[key].reshape(-1,*s[2:])
        B, N = x['id_type'].shape
        id_embed = self.id_embedder(x['id_type'])
        occluded_embed = self.occluded_embedder(x['occluded'])
        # get all the inputs for fourier embedding
        fourier_input = torch.cat((x['truncated'].unsqueeze(-1),
                                   x['alpha'].unsqueeze(-1),
                                   x['bbox'],
                                   x['dimensions'],
                                   x['locations'],
                                   x['rotation_y'].unsqueeze(-1),), dim=-1)
        fourier_embed = self.fourier_embedder(fourier_input).view(B, N, -1)
        all_embed = torch.cat((fourier_embed, id_embed, occluded_embed), dim=-1)
        out = self.mlp(all_embed)
        if is_clip:
            out = out.reshape(*original_shape[:2],*out.shape[1:])
        return out