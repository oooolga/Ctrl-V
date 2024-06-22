import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing

from PIL import Image
import numpy as np
import os

from einops import rearrange


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = nn.Parameter(temperature ** ( torch.arange(num_freqs) / num_freqs ), requires_grad=False)

    @torch.no_grad()
    def __call__(self, x):
        "x: arbitrary shape of tensor. dim: cat dim"
        temp = torch.einsum('...,k->...k', x, self.freq_bands)
        return torch.cat((torch.sin(temp), torch.cos(temp)), -1)

def tokenize_captions(batch_prompts, tokenizer):
    inputs = tokenizer(
        batch_prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def get_dataloader(dset_root, dset_name, if_train, batch_size, num_workers, data_type='image', clip_length=10,
                   collate_fn=None, use_default_collate=True, tokenizer=None, shuffle=True, if_return_bbox_im=False,
                   train_H=None, train_W=None, use_segmentation=False, use_preplotted_bbox=True, if_last_frame_traj=False):
    if dset_name.lower() == 'kitti':
        from ctrlv.datasets import KittiDataset
        dset = KittiDataset(root=dset_root, train=if_train, data_type=data_type, clip_length=clip_length, if_return_bbox_im=if_return_bbox_im,
                            train_H=train_H, train_W=train_W) #, use_preplotted_bbox=use_preplotted_bbox)
    elif dset_name.lower() == 'vkitti':
        from ctrlv.datasets import VKittiDataset
        dset = VKittiDataset(root=dset_root, train=if_train, data_type=data_type, clip_length=clip_length, if_return_bbox_im=if_return_bbox_im,
                             train_H=train_H, train_W=train_W, use_preplotted_bbox=use_preplotted_bbox)
    elif dset_name.lower() == 'mkitti':
        from ctrlv.datasets import MergedKittiDataset
        dset = MergedKittiDataset(root=dset_root, train=if_train, data_type=data_type, clip_length=clip_length, if_return_bbox_im=if_return_bbox_im,
                                  train_H=train_H, train_W=train_W, use_preplotted_bbox=use_preplotted_bbox)
    elif dset_name.lower() == 'bdd100k':
        from ctrlv.datasets import BDD100KDataset
        if use_segmentation:
            use_preplotted_bbox = True
        dset = BDD100KDataset(root=dset_root, train=if_train, data_type=data_type, clip_length=clip_length, if_return_bbox_im=if_return_bbox_im,
                              train_H=train_H, train_W=train_W, use_segmentation=use_segmentation, use_preplotted_bbox=use_preplotted_bbox)
        dset.set_if_last_frame_trajectory(if_last_frame_traj)
    elif dset_name.lower() == 'davis':
        from ctrlv.datasets import DAVISDataset
        dset = DAVISDataset(root=dset_root, train=if_train, data_type=data_type, clip_length=clip_length, if_return_bbox_im=if_return_bbox_im,
                            train_H=train_H, train_W=train_W, use_preplotted_bbox=True)
    else:
        raise NotImplementedError("Dataset not implemented")

    if use_default_collate:

        from ctrlv.datasets import kitti_collate_fn, kitti_clip_collate_fn, kitti_clip_with_bbox_collate_fn
        if tokenizer is None:
            tokenize_fn = None
        else:
            tokenize_fn = lambda x: tokenize_captions(x, tokenizer)
        if data_type == 'image':
            collate_fn = lambda x: kitti_collate_fn(x, tokenize_fn)
        elif not if_return_bbox_im:
            collate_fn = lambda x: kitti_clip_collate_fn(x, tokenize_fn)
        else:
            collate_fn = lambda x: kitti_clip_with_bbox_collate_fn(x, tokenize_fn)
    return dset, torch.utils.data.DataLoader(
        dset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )



def encode_video_image(pixel_values, feature_extractor, weight_dtype, image_encoder):

    def resize_video_image(pixel_values):
        # import torchvision
        # # resize_fn = torchvision.transforms.Resize((224, 224), antialias=True)
        # # pixel: [-1, 1]
        # pixel_values = resize_fn(pixel_values)
        
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # # unnormalize
        pixel_values = (pixel_values+1.0)*0.5
        pixel_values = torch.clamp(pixel_values, min=0., max=1.)
        return pixel_values

    pixel_values = resize_video_image(pixel_values=pixel_values)
    
    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(device=image_encoder.device).to(dtype=weight_dtype)
    image_embeddings = image_encoder(pixel_values).image_embeds
    return image_embeddings

def get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        unet
    ):
    """"https://github.com/huggingface/diffusers/blob/56bd7e67c2e01122cc93d98f5bd114f9312a5cce/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L215"""
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def rescale_bbox(bbox, image_size=(1242, 375), target_size=(1, 1)):
    """Rescales bounding boxes to the target size."""

    orig_shape = bbox.shape
    bbox = bbox.clone().reshape(-1, 4)
    bbox[:, 0] = bbox[:, 0] * target_size[0] / image_size[0]
    bbox[:, 1] = bbox[:, 1] * target_size[1] / image_size[1]
    bbox[:, 2] = bbox[:, 2] * target_size[0] / image_size[0]
    bbox[:, 3] = bbox[:, 3] * target_size[1] / image_size[1]

    return bbox.reshape(*orig_shape)

def get_fourier_embeds_from_boundingbox(objects, image_size=(1242, 375), dropout_prob=0., generator=None, embed_dim=8):
    bbox = objects['bbox']
    bits = 4
    # resize bbox to [0, 1]
    rescaled_bbox = rescale_bbox(bbox, 
                                 image_size=image_size,
                                 target_size=(1, 1))
    # rearrange boxes data
    object_boxes = torch.cat((
        objects['truncated'].unsqueeze(-1), 
        objects['alpha'].unsqueeze(-1), 
        rescaled_bbox, 
        objects['dimensions'], 
        objects['locations'],
        objects['rotation_y'].unsqueeze(-1)), dim=-1)
    batch_size, num_frames, num_boxes, _ = object_boxes.shape
    object_boxes = object_boxes.reshape(batch_size*num_frames, num_boxes, -1)
    
    object_ids = to_binary(objects['track_id'], bits=bits).to(rescaled_bbox.dtype)
    object_ids = F.normalize(object_ids, p=2, dim=-1).reshape(batch_size*num_frames, num_boxes, -1)

    frame_ids = torch.arange(num_frames)/float(num_frames)
    frame_ids = frame_ids[None, :, None, None].repeat(batch_size, 1, num_boxes, 1).to(device=rescaled_bbox.device, dtype=rescaled_bbox.dtype)
    frame_ids = frame_ids.reshape(batch_size*num_frames, num_boxes, -1)

    type_ids = to_binary(objects['id_type'], bits=bits).to(rescaled_bbox.dtype)
    type_ids = F.normalize(type_ids, p=2, dim=-1).reshape(batch_size*num_frames, num_boxes, -1)
    
    object_boxes = torch.cat([object_boxes, object_ids, frame_ids, type_ids], dim=-1)
    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=rescaled_bbox.device, dtype=rescaled_bbox.dtype)
    emb = emb * object_boxes.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, num_boxes, embed_dim * 2 * (13+2*bits+1))

    # make paddings
    num_objects = torch.LongTensor(objects['num_objects']).unsqueeze(-1)
    buff = torch.arange(num_boxes)[None, None, :].repeat(batch_size, num_frames, 1)
    tmp = (buff < num_objects).unsqueeze(-1).to(device=rescaled_bbox.device)
    emb = torch.where(tmp, emb, torch.zeros_like(emb))

    # dropout bbox
    random_p = torch.rand(batch_size, num_frames, num_boxes, 1, device=rescaled_bbox.device, generator=generator)
    dropout_mask = random_p < dropout_prob
    emb = torch.where(dropout_mask, torch.zeros_like(emb), emb)
    return emb

def to_binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def get_first_training_sample(batch, dataset):
    clip = batch['clips'][:1,::]
    gt_clip = clip[0,::].detach().cpu()
    gt_clip_np = dataset.revert_transform_no_resize(gt_clip).detach().cpu().numpy()*255
    gt_clip_np = gt_clip_np.astype(np.uint8)
    clip_idx = batch['indices'][0]
    image_init_f = dataset.get_frame_file_by_index(clip_idx, 0)
    image_init = Image.open(image_init_f)
    if not dataset.if_return_bbox_im:
        _, gt_labels, _, cam_to_img, _ = dataset.__getitem__(clip_idx, return_calib=True)

        sample = dict(
                gt_clip             =           gt_clip,
                objects_tensors     =           batch['objects'],
                image_init          =           image_init.resize((dataset.train_W, dataset.train_H)),
                gt_labels           =           gt_labels,
                cam_to_img          =           cam_to_img,
            )
    else:
        _, gt_labels, _, cam_to_img, _, bbox_img = dataset.__getitem__(clip_idx, return_calib=True)
        bbox_img_np = dataset.revert_transform_no_resize(bbox_img).detach().cpu().numpy()*255
        bbox_img_np = bbox_img_np.astype(np.uint8)
        bbox_img = bbox_img.detach().cpu()
        bbox_init_f = dataset.get_bbox_image_file_by_index(image_file=image_init_f)
        bbox_init = Image.open(bbox_init_f).convert('RGB')

        sample = dict(
                gt_clip             =           gt_clip,
                gt_clip_np          =           gt_clip_np,
                objects_tensors     =           batch['objects'],
                image_init          =           image_init.resize((dataset.train_W, dataset.train_H)),
                gt_labels           =           gt_labels,
                cam_to_img          =           cam_to_img,
                bbox_img            =           bbox_img,
                bbox_img_np         =           bbox_img_np,
                bbox_init           =           bbox_init.resize((dataset.train_W, dataset.train_H))
            )
    return sample
    

def get_n_training_samples(data_loader, n_samples):
    samples = []

    dataset = data_loader.dataset
    for i, batch in enumerate(data_loader):
        if i >= n_samples:
            break
        sample = get_first_training_sample(batch, dataset)
        samples.append(sample)
        
    return samples

def eval_samples_generator(data_loader):
    assert data_loader.batch_size == 1
    for i, batch in enumerate(data_loader):
        sample = get_first_training_sample(batch, data_loader.dataset)
        yield sample

def eval_demo_samples_generator(pkl_files):
    import pickle
    for pkl_f in pkl_files:
        print(pkl_f)
        with open(pkl_f, 'rb') as f:
            yield pickle.load(f)