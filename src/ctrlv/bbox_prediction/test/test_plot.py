# print("Importing")
# from sd3d.utils import get_dataloader, get_n_training_samples, plot_3d_bbox, save_image
# print("Loading")
# dataset, dataloader = get_dataloader('kitti', if_train=True, batch_size=1, num_workers=1, 
#                             data_type='clip', clip_length=25, use_default_collate=True, tokenizer=None, shuffle=False)

# print("Dataset size:", len(dataset))

# # dict_keys(['gt_clip', 'objects_tensors', 'image_init', 'gt_labels', 'cam_to_img'])

# demo_samples = get_n_training_samples(dataloader, 1)


# test_batch = next(iter(dataloader))
# gt_img, labels, prompt, cam_to_img, _ = dataloader.dataset.__getitem__(test_batch['indices'][0], return_calib=True)
# print("Got data")

# i = 0 
# img_plot = plot_3d_bbox(gt_img[i], labels[i], cam_to_img, is_gt=True)

# save_image(img_plot, "test_image.png")


# # for sample_i, sample in enumerate(demo_samples):
# #     print(sample.keys())
# #     gt_img, labels, prompt, cam_to_img, last = sample
# #     print("Done")

# print("Done")

# from sd3d.utils import get_dataloader, plot_3d_bbox, save_image
# import torch, os
# from tqdm import tqdm

# # print('Start preprocessing kitti dataset.')
# dataset, _ = get_dataloader("/network/scratch/x/xuolga/Datasets", 'kitti', if_train=True, batch_size=1, num_workers=0, 
#              data_type='image', clip_length=25, use_default_collate=True, tokenizer=None, shuffle=False)


# bbox_dir = os.path.join(dataset.root, dataset.version, dataset._location, dataset.TO_BBOX_DIR)

# image_dir = os.path.join(dataset.root, dataset.version, dataset._location, dataset.TO_IMAGE_DIR)
# for scene in sorted(os.listdir(image_dir)):
#     scene_dir = os.path.join(bbox_dir, scene)
#     os.makedirs(scene_dir, exist_ok=True)

# for i in tqdm(range(len(dataset))):
    
#     if i % 10 != 0: continue
    
#     pixel_value, objects, _, cam_to_img, _ = dataset.__getitem__(i, return_prompt=True, return_calib=True, return_index=True)
#     pixel_value = dataset.revert_transform(pixel_value)
#     canvas = torch.zeros_like(pixel_value)
#     bbox_im = plot_3d_bbox(canvas, objects, cam_to_img, plot_2d_bbox=True)
#     # save_path = dataset.get_image_file_by_index(i).replace(dataset.TO_IMAGE_DIR, dataset.TO_BBOX_DIR)
#     save_image(bbox_im, f"test_image{i}.png", verbose=False)
#     print("saved one image")

# print('Finished preprocessing kitti dataset.')

# val_dataset, val_dataloader = get_dataloader("/network/scratch/x/xuolga/Datasets", 'kitti', if_train=False, batch_size=1, num_workers=0, 
#                 data_type='clip', clip_length=25, use_default_collate=True, tokenizer=None, shuffle=False)

# train_dataset, train_dataloader = get_dataloader("/network/scratch/x/xuolga/Datasets", 'kitti', if_train=True, batch_size=1, num_workers=0, 
#                 data_type='clip', clip_length=25, use_default_collate=True, tokenizer=None, shuffle=False)


# while True:
#     train_batch = next(iter(train_dataloader))
#     val_batch = next(iter(val_dataloader))
#     print("---")

import torch

# # Example tensor
# ids = torch.tensor([[[0, 1, 3, 64, 65, 0, 0, 0, 0, 0],
#                      [0, 1, 3, 64, 65, 0, 0, 0, 0, 0]],
#                     [[0, 2, 4, 66, 67, 0, 0, 0, 0, 0],
#                      [0, 2, 4, 66, 67, 0, 0, 0, 0, 0]]], dtype=torch.int64)  # shape: (batch, timesteps, max_num_agents)

# # Create a mask for all zero values
# zero_mask = ids == 0

# # Create a mask for zero values in the first position of the last dimension
# first_position_mask = torch.zeros_like(ids, dtype=torch.bool)
# first_position_mask[:, :, 0] = ids[:, :, 0] == 0

# # Combine the masks: zeros that are not in the first position
# final_mask = zero_mask & ~first_position_mask

# # Set the selected zero values to -1
# ids[final_mask] = -1

# # Print the modified tensor
# print(ids)

from sd3d.utils import plot_3d_bbox
import imageio
def create_video_from_numpy_array(array, output_filename, fps=30):
    # Get the dimensions of the array
    num_frames, height, width, channels = array.shape

    # Create a video writer object
    with imageio.get_writer(output_filename, fps=fps) as writer:
        for i in range(num_frames):
            frame = array[i]
            # Append each frame to the video
            writer.append_data(frame)
    
    print(f'Video saved as {output_filename}')


H, W = 320, 512

import numpy as np
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((W, H)),
    transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)), # map from [0, 255] to [-1,1]
 ])

revert_transform_no_resize = transforms.Compose([
    transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
])

pos = [
    torch.tensor([10, 10, W//2, H//2]),
    torch.tensor([0, 0, .5, .5]),
    torch.tensor([0, 0, 512//2, 320//2]),
]


img_stack = torch.zeros([25, H, W, 3])
for t in range(25):
    labels = []
    for i in range(1):
        labels.append({
            "bbox": pos[i],
            "id_type": i,
            "trackID": i
        })


    canvas = torch.zeros((3, H, W))  # Black background
    bbox_im = plot_3d_bbox(canvas, labels, None, plot_2d_bbox=True)

    img_stack[t] = torch.tensor(bbox_im)#.permute(2, 0, 1)

bbox_vid = transform(img_stack.permute(0, 3, 2, 1))
bbox_vid_np = revert_transform_no_resize(bbox_vid).detach().cpu().numpy()*255
bbox_vid_np = bbox_vid_np.astype(np.uint8).transpose([0, 3, 2, 1])

# log_dict["predicted_rollout"].append(wandb.Video(bbox_vid_np, fps=self.cfg.fps))
# log_dict[f"gt_bbox_frames_{sample_i}"].append(wandb.Video(sample['bbox_img_np'], fps=self.cfg.video_fps))
create_video_from_numpy_array(bbox_vid_np, f"TEST_VID.mp4", fps=7)
# create_video_from_numpy_array(sample['bbox_img_np'], f"gt_video_{sample_i}.mp4", fps=7)