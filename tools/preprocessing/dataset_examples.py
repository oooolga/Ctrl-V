from ctrlv.utils.util import get_dataloader

# replace the root to the directory where your dataset folders are located
root = "/network/scratch/x/xuolga/Datasets/"
# TO LOAD TOKENIZER FOR PROMPTS
# you may also set tokenizer=None in get_dataloader to not tokenize the prompts
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained(
        'stabilityai/stable-diffusion-2-1', subfolder='tokenizer'
    )



# TO LOAD KITTI CLIP DATASET
## kitti example; for this moment, if_train has to be True
kitti_image_dset, kitti_image_loader = get_dataloader(root, 'bdd100k', if_train=True, batch_size=10, num_workers=1, data_type='clip', clip_length=25, use_default_collate=True, tokenizer=tokenizer, shuffle=True)
batch = next(iter(kitti_image_loader))
import wandb
wandb.init(project="ctrlv",name="fps_test")


import pdb; pdb.set_trace()


# TO LOAD KITTI CLIP DATASET
## kitti example; for this moment, if_train has to be True
kitti_image_dset, kitti_image_loader = get_dataloader(root, 'kitti', if_train=True, batch_size=10, num_workers=1, data_type='image', use_default_collate=True, tokenizer=tokenizer, shuffle=True)
batch = next(iter(kitti_image_loader))
## You can use object net to encode the objects in the frame
from ctrlv.models import KittiObjectNet
object_net = KittiObjectNet(out_dim=1024)
object_emb = object_net(batch["objects"])

import pdb; pdb.set_trace()
del kitti_image_dset, kitti_image_loader, batch, object_emb

# TO LOAD KITTI CLIP DATASET
kitti_video_dset, kitti_video_loader = get_dataloader(root, 'kitti', if_train=True, batch_size=10, num_workers=1, data_type='clip', use_default_collate=True, tokenizer=tokenizer, shuffle=True)
batch = next(iter(kitti_video_loader))
import pdb; pdb.set_trace()
del kitti_video_dset, kitti_video_loader, batch

# TO LOAD VKITTI CLIP DATASET
kitti_video_dset, kitti_video_loader = get_dataloader(root, 'vkitti', if_train=True, batch_size=10, num_workers=1, data_type='clip', use_default_collate=True, tokenizer=tokenizer, shuffle=True)
batch = next(iter(kitti_video_loader))
import pdb; pdb.set_trace()
del kitti_video_dset, kitti_video_loader, batch

# TO LOAD MKITTI CLIP DATASET
kitti_video_dset, kitti_video_loader = get_dataloader(root, 'mkitti', if_train=True, batch_size=10, num_workers=1, data_type='clip', use_default_collate=True, tokenizer=tokenizer, shuffle=True)
batch = next(iter(kitti_video_loader))
import pdb; pdb.set_trace()
del kitti_video_dset, kitti_video_loader, batch