from ctrlv.utils import get_dataloader, eval_samples_generator, plot_3d_bbox, save_image
import torch
import pickle, os
from torchvision import transforms
import numpy as np
import random, copy

# replace the root to the directory where your dataset folders are located
root = '/network/scratch/x/xuolga/Datasets'
demo_path = './demos/bbox_demo'
dataset, data_loader = get_dataloader(root, 'bdd100k', if_train=True, batch_size=1, num_workers=1, clip_length=25,
                                      data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True, if_return_bbox_im=True)
sample_generator = eval_samples_generator(data_loader)

transform = transforms.Compose([
    transforms.Resize((dataset.train_H, dataset.train_W)),
    transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5)), # map from [0, 255] to [-1,1]
 ])

def check_bounds(bbox, H, W):
    bbox[0] = max(min(bbox[0], W*0.8), 0)
    bbox[1] = max(min(bbox[1], H*0.8), 0)
    bbox[2] = max(min(bbox[2], W), bbox[0]+1)
    bbox[3] = max(min(bbox[3], H), bbox[1]+1)
    return bbox


for sample_i, sample in enumerate(sample_generator):

    os.makedirs(os.path.join(demo_path, f'fake_bbox_{sample_i}'), exist_ok=True)
    os.makedirs(os.path.join(demo_path, f'real_bbox_{sample_i}'), exist_ok=True)

    for i in range(25):
        canvas = torch.zeros((3, dataset.orig_H, dataset.orig_W))
        if i == 0:
            bbox_im = plot_3d_bbox(canvas, sample['gt_labels'][i], None, plot_2d_bbox=True)
            save_path = os.path.join(demo_path, f'fake_bbox_{sample_i}', f'bbox_im_{sample_i}_{i:02d}.png')
            save_image(bbox_im, save_path, verbose=False)
            fake_bbox = torch.tensor(bbox_im).permute(2,0,1).unsqueeze(0)
        elif i == 24:
            temp = copy.deepcopy(sample['gt_labels'][i])
            for j in range(len(temp)-1, -1, -1):
                prob = random.uniform(0, 1)
                if prob < 0.25:
                    temp.pop(j)
                elif prob < 0.3:
                    temp[j]['trackID'] = random.randint(0, temp[j]['trackID']-50)
            
            for j in range(len(temp)):
                temp[j]['bbox'][0] += random.randint(-dataset.orig_W//5, dataset.orig_W//5)
                temp[j]['bbox'][1] += random.randint(-dataset.orig_H//5, dataset.orig_H//5)
                ratio = random.uniform(0.75, 1.1)
                temp[j]['bbox'][2] = temp[j]['bbox'][0] + ratio*(sample['gt_labels'][i][j]['bbox'][2] - sample['gt_labels'][i][j]['bbox'][0]) + random.randint(-5, 5)
                temp[j]['bbox'][3] = temp[j]['bbox'][1] + ratio*(sample['gt_labels'][i][j]['bbox'][3] - sample['gt_labels'][i][j]['bbox'][1]) + random.randint(-5, 5)
                temp[j]['bbox'] = check_bounds(temp[j]['bbox'], dataset.orig_H, dataset.orig_W)
            
            
            bbox_im = plot_3d_bbox(canvas, temp, None, plot_2d_bbox=True)
            fake_bbox = torch.cat((fake_bbox, torch.tensor(bbox_im).permute(2,0,1).unsqueeze(0)), dim=0)
            save_path = os.path.join(demo_path, f'fake_bbox_{sample_i}', f'bbox_im_{sample_i}_{i:02d}.png')
            save_image(bbox_im, save_path, verbose=False)
            canvas = torch.zeros((3, dataset.orig_H, dataset.orig_W))
            bbox_im = plot_3d_bbox(canvas, sample['gt_labels'][i], None, plot_2d_bbox=True)
        else:
            save_path = os.path.join(demo_path, f'fake_bbox_{sample_i}', f'bbox_im_{sample_i}_{i:02d}.png')
            temp = plot_3d_bbox(canvas, [], None)
            fake_bbox = torch.cat((fake_bbox, torch.tensor(temp).permute(2,0,1).unsqueeze(0)), dim=0)
            save_image(temp, save_path, verbose=False)
            bbox_im = plot_3d_bbox(canvas, sample['gt_labels'][i], None, plot_2d_bbox=True)
        save_path = os.path.join(demo_path, f'real_bbox_{sample_i}', f'bbox_im_{sample_i}_{i:02d}.png')
        save_image(bbox_im, save_path, verbose=False)
    
    dump_file = os.path.join(demo_path, 'original_sample_%d.pkl' % sample_i)
    pickle.dump(sample, open(dump_file, 'wb'))
    
    bbox_img = transform(fake_bbox)
    bbox_img_np = dataset.revert_transform_no_resize(bbox_img).detach().cpu().numpy()*255
    bbox_img_np = bbox_img_np.astype(np.uint8)
    
    sample['bbox_img'] = bbox_img
    sample['bbox_img_np'] = bbox_img_np
    dump_file = os.path.join(demo_path, 'fake_bbox_%d.pkl' % sample_i)
    pickle.dump(sample, open(dump_file, 'wb'))
    if sample_i >= 2:
        break