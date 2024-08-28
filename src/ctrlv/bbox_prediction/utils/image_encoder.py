import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import PIL.Image
import numpy as np

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder

from ctrlv.bbox_prediction.utils import MLPLayer


class ImageEncoder(nn.Module):

    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()

        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        revision = None
        pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid-xt"

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=revision, variant="fp16",
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="image_encoder", revision=revision, variant="fp16"
        )

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.project_clip_embeddings = nn.Sequential(  # hidden_dim = self.cfg.hidden_dim
            MLPLayer(1024, self.cfg.hidden_dim, self.cfg.hidden_dim)
            # nn.BatchNorm1d(num_features=1)
        ).to(self.device)

        in_channels, output_size, out_channels = 4, 16, 32
        self.project_vae_embeddings = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(output_size*2, output_size*2)),
            
            nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(output_size, output_size)),
            
            nn.Flatten(start_dim=2), # Flatten spatial dimension, keep channels
            MLPLayer(output_size**2, self.cfg.hidden_dim, self.cfg.hidden_dim),
            # nn.BatchNorm1d(num_features=self.cfg.hidden_dim)
        ).to(self.device)

        # self.image_embedding_layer = MLPLayer(self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.hidden_dim)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

    def forward(self, 
                 image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor], 
                 image_size = (320, 512)):
        
        # TODO: Define device
        num_videos_per_prompt = 1

        # CLIP embeddings
        image_embeddings = self._encode_image(
            image, 
            self.device, 
            num_videos_per_prompt, 
            do_classifier_free_guidance=False
        )
        
        height, width = image_size
        image = self.image_processor.preprocess(image, height=height, width=width).to(self.device)
        # image = self.image_processor.preprocess(image).to(device) # Using default width and height

        # TODO: Can we just skip adding noise to the image?
        # noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        # image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=self.device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=False,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        
        
        # Reshape (and possibly run through another layer) before returning
        clip_embeddings = self.project_clip_embeddings(image_embeddings)
        vae_embeddings = self.project_vae_embeddings(image_latents)
        
        output_embeddings = torch.cat([clip_embeddings, vae_embeddings], dim=1)
        # output_embeddings = self.image_embedding_layer(output_embeddings)

        return output_embeddings  # [batch_size, features, hidden_dim]


    def _encode_image(
            self,
            image: PipelineImageInput,
            device: Union[str, torch.device],
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
        ) -> torch.FloatTensor:
            dtype = next(self.image_encoder.parameters()).dtype

            if not isinstance(image, torch.Tensor):
                image = self.image_processor.pil_to_numpy(image)
                image = self.image_processor.numpy_to_pt(image)

                # We normalize the image before resizing to match with the original implementation.
                # Then we unnormalize it after resizing.
                image = image * 2.0 - 1.0
                image = _resize_with_antialiasing(image, (224, 224))
                image = (image + 1.0) / 2.0

            # Normalize the image with for CLIP input
            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeddings = self.image_encoder(image).image_embeds
            image_embeddings = image_embeddings.unsqueeze(1)

            # duplicate image embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = image_embeddings.shape
            image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
            image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

            if do_classifier_free_guidance:
                negative_image_embeddings = torch.zeros_like(image_embeddings)

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

            return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents



def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


if __name__ == '__main__':
    image_encoder = ImageEncoder()