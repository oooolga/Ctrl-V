import numpy as np
import torch
import scipy.linalg
from typing import Tuple
import torch.nn.functional as F
import math
from torchvision import transforms

"""
Copy-pasted from Copy-pasted from https://github.com/NVlabs/stylegan2-ada-pytorch
"""
import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Dict


def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

"""
Modified from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
class FVD:
    def __init__(self, device,
                 detector_url='https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1',
                 rescale=False, resize=False, return_features=True):
        
        self.device = device
        self.detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
    
    def to_device(self, device):
        self.device = device
        self.detector = self.detector.to(self.device)
    
    def _compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0) # [d]
        sigma = np.cov(feats, rowvar=False) # [d, d]
        return mu, sigma
    
    def preprocess_videos(self, videos, resolution=224, sequence_length=None):
        
        b, t, c, h, w = videos.shape
        
        # temporal crop
        if sequence_length is not None:
            assert sequence_length <= t
            videos = videos[:, :sequence_length, ::]
        
        # b*t x c x h x w
        videos = videos.reshape(-1, c, h, w)
        if c == 1:
            videos = torch.cat([videos, videos, videos], 1)
            c = 3
        
        # scale shorter side to resolution
        scale = resolution / min(h, w)
        # import pdb; pdb.set_trace()
        if h < w:
            target_size = (resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), resolution)
        
        videos = F.interpolate(videos, size=target_size).clamp(min=-1, max=1)
        
        # center crop
        _, c, h, w = videos.shape
        
        h_start = (h - resolution) // 2
        w_start = (w - resolution) // 2
        videos = videos[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
        
        # b, c, t, w, h
        videos = videos.reshape(b, t, c, resolution, resolution).permute(0, 2, 1, 3, 4)
        
        return videos.contiguous()
    
    @torch.no_grad()
    def evaluate(self, video_fake, video_real, res=224):
        
        video_fake = self.preprocess_videos(video_fake,resolution=res)
        video_real = self.preprocess_videos(video_real,resolution=res)
        feats_fake = self.detector(video_fake, **self.detector_kwargs).cpu().numpy()
        feats_real = self.detector(video_real, **self.detector_kwargs).cpu().numpy()
        
        mu_gen, sigma_gen = self._compute_stats(feats_fake)
        mu_real, sigma_real = self._compute_stats(feats_real)
        
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return fid

def evaluate_vids(vid_dir):
    
    f_gt_vid = []
    f_gen_vid = []
    for fname in os.listdir(vid_dir):
        if fname.startswith('generated_videos'):
            f_gen_vid.append(fname)
        if fname.startswith('gt_videos'):
            f_gt_vid.append(fname)
    f_gt_vid.sort(key=sort_name)
    f_gen_vid.sort(key=sort_name)
    assert(len(f_gt_vid) == len(f_gen_vid))
    # f_gt_vid = f_gt_vid[:201]
    # f_gen_vid = f_gen_vid[:201]
    
    # import pdb; pdb.set_trace()
    # all_gt = np.zeros((201,16,3,320,512))
    # all_gen = np.zeros((201,16,3,320,512))
    all_gt = np.zeros((201,25,3,256,410))
    all_gen = np.zeros((201,25,3,256,410))
    valid = 0
    for idx,(fgen,fgt) in enumerate(zip(f_gen_vid, f_gt_vid)):
        if valid == 201:
            break
        print(idx)
        assert (sort_name(fgen) == sort_name(fgt))
        
        with Image.open(os.path.join(vid_dir,fgt)) as im:
            gt_vid = load_frames(im,size=(410,256))
        with Image.open(os.path.join(vid_dir,fgen)) as im:
            gen_vid = load_frames(im,size=(410,256))
        if gt_vid.shape[0] < 25 or gen_vid.shape[0] < 25:
            continue

        gt_vid = np.expand_dims(gt_vid,0).transpose(0,1,4,2,3)
        gen_vid = np.expand_dims(gen_vid,0).transpose(0,1,4,2,3)

        all_gt[valid] = gt_vid[:,:25,::]
        all_gen[valid] = gen_vid[:,:25,::]
        valid += 1

    all_gt = torch.from_numpy(all_gt).cuda().float()
    all_gt /= 255/2.0
    all_gt -= 1.0
    all_gen = torch.from_numpy(all_gen).cuda().float()
    all_gen /= 255/2.0
    all_gen -= 1.0
    # import pdb; pdb.set_trace()
    fvd = FVD(device ='cuda')
    fvd_score = fvd.evaluate(all_gt,all_gen)
    # fvd_score = fvd.evaluate(all_gt[:134],all_gen[:134])
    del fvd

    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.cuda()
    lpips_score = 0
    for idx in range(all_gen.shape[0]): 
    # for idx in range(134): 
        lpips_score += loss_fn_alex(all_gt[idx],all_gen[idx])/all_gen.shape[0]
    lpips_score = lpips_score.mean().item()
    del loss_fn_alex
    # import pdb; pdb.set_trace()
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    all_gen = all_gen.detach().cpu().numpy()
    all_gt = all_gt.detach().cpu().numpy()

    ssim_score_vid = np.zeros(201)
    ssim_score_image = np.zeros((201,25))
    psnr_score_vid = np.zeros(201)
    psnr_score_image = np.zeros((201,25))
    psnr_score_all = psnr(all_gt,all_gen)
    for vid_idx in range(all_gen.shape[0]):
        print(f'vid idx: {vid_idx}')
        for f_idx in range(all_gen.shape[1]):
            img_gt = all_gt[vid_idx,f_idx]
            img_gen = all_gen[vid_idx,f_idx]
            data_range = max(img_gt.max(),img_gen.max()) - min(img_gt.min(),img_gen.min())
            ssim_score_image[vid_idx,f_idx] = ssim(img_gt,img_gen,channel_axis=0,data_range=data_range,gaussian_weights=True,sigma=1.5)
            psnr_score_image[vid_idx,f_idx] = psnr(img_gt,img_gen,data_range=data_range)

        vid_gt = all_gt[vid_idx]
        vid_gen = all_gen[vid_idx]
        data_range = max(vid_gt.max(),vid_gen.max()) - min(vid_gt.min(),vid_gen.min())
        ssim_score_vid[vid_idx] = ssim(vid_gt,vid_gen,channel_axis=1,data_range=data_range,gaussian_weights=True,sigma=1.5)
        psnr_score_vid[vid_idx] = psnr(vid_gt,vid_gen,data_range=data_range)
    
    ssim_score_image_error = np.sqrt(((ssim_score_image - ssim_score_image.mean())**2).sum()/200)
    psnr_score_image_error = np.sqrt(((psnr_score_image - psnr_score_image.mean())**2).sum()/200)

    print(f'fvd_score: {fvd_score}')
    print(f'lpips_score: {lpips_score}')
    # print(f'ssim_score_vid: {ssim_score_vid.mean()}')
    print(f'ssim_score_image: {ssim_score_image.mean()}')
    print(f'ssim_score_image_error: {ssim_score_image_error}')
    # print(f'psnr_score_vid: {psnr_score_vid.mean()}')
    print(f'psnr_score_image: {psnr_score_image.mean()}')
    print(f'psnr_score_image_error: {psnr_score_image_error}')
    # print(f'psnr_score_all: {psnr_score_all}')
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    # from ctrlv.utils.util import get_dataloader
    # root = "/network/scratch/x/xuolga/Datasets/"
    # kitti_image_dset, kitti_image_loader = get_dataloader(root, 'bdd100k', if_train=True, batch_size=10, num_workers=1, data_type='clip', clip_length=25, use_default_collate=True, tokenizer=None, shuffle=True)
    # batch = next(iter(kitti_image_loader))
    # fvd = FVD(device = 'cpu')
    # vid = batch['clips'].to('cpu')
    # score = fvd.evaluate(vid,vid)
    # import pdb; pdb.set_trace()
    
    import os
    import sys
    from PIL import Image, ImageSequence
    import numpy as np

    vid_dir = sys.argv[1]

    def sort_name(text):
        text = text[text.find('videos_')+len('videos_'):]
        num = text[:text.find('_')]
        return int(num)
    

    def load_frames(image: Image, mode='RGB',size=(256,256)):
        return np.array([
            np.array(frame.resize(size).convert(mode))
            for frame in ImageSequence.Iterator(image)
        ])

    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240531_023312-8s6ospag/files/media/videos/'
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240601_175458-0v85adgs/files/media/videos/'
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/vkitti_ctrlv_240510_150856/wandb/run-20240531_212327-sehj32yk/files/media/videos/'
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/kitti_ctrlv_240510_141159/wandb/run-20240601_204019-649cucdi/files/media/videos/'
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/vkitti_ctrlv_240510_150856/wandb/run-20240602_212229-tso8rfs4/files/media/videos/'
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/kitti_ctrlv_240510_141159/wandb/run-20240603_030728-5d9g8nol/files/media/videos/'

    #kitti baseline
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/kitti_video_240603_012345/wandb/run-20240605_113611-nr6axueg/files/media/videos/'

    #vkitti baseline
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/vkitti_video_240602_014243/wandb/run-20240606_095618-h2topzfq/files/media/videos/'

    #bdd baseline
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/bdd100k_video_240525_163350/wandb/run-20240605_120657-rmzcct7y/files/media/videos/'

    # last frame point
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240607_200141-56arymuj/files/media/videos/'

    # kitti 1-to-1
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/kitti_ctrlv_240510_141159/wandb/run-20240611_002604-xm6jteh5/files/media/videos/'

    # bdd 1-to-1
    # vid_dir = '/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240611_003700-xb81s7c1/files/media/videos/'


    # for vid_dir in ['/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240531_023312-8s6ospag/files/media/videos/','/network/scratch/x/xuolga/Results/sd3d/bdd100k_ctrlv_240511_200727/wandb/run-20240601_175458-0v85adgs/files/media/videos/','/network/scratch/x/xuolga/Results/sd3d/vkitti_ctrlv_240510_150856/wandb/run-20240531_212327-sehj32yk/files/media/videos/','/network/scratch/x/xuolga/Results/sd3d/kitti_ctrlv_240510_141159/wandb/run-20240601_204019-649cucdi/files/media/videos/']:
    evaluate_vids(vid_dir)
    
    import pdb; pdb.set_trace()