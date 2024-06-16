### This page is currently under construction

<p align="center">
<a href="https://oooolga.github.io/ctrl-v.github.io/">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./statics/logo_darkmode.png" height="80">
        <source media="(prefers-color-scheme: light)" srcset="./statics/logo.png" height="80">
        <img alt="logo" src="./statics/logo.png" height="80">
    </picture>
</a>
</p>

# Ctrl-V: Higher Fidelity Video Generation with Bounding-Box Controlled Object Motion

<p align="left">
<a href="https://arxiv.org/abs/2406.05630" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2406.05630-b31b1b.svg?style=flat" /></a>
<a href="https://oooolga.github.io/ctrl-v.github.io/" alt="webpage">
    <img src="https://img.shields.io/badge/Webpage-Ctrl_V-darkviolet" /></a>
<img src="https://img.shields.io/github/license/oooolga/Ctrl-V" />
<a href="https://paperswithcode.com/paper/ctrl-v-higher-fidelity-video-generation-with">
    <img alt="Static Badge" src="https://img.shields.io/badge/paper_with_code-link-turquoise?logo=paperswithcode" />
</a>
<p align="center">
<picture>
  <source srcset="./statics/CtrlV_teaserv2.png" media="(prefers-color-scheme: light)">
  <source srcset="./statics/CtrlV_teaser_darkmode.png" media="(prefers-color-scheme: dark)">
  <img src="./statics/CtrlV_teaserv2.png">
</picture>
</p>

# Quick Start
### Download requirements
```
pip install -r requirment.txt
```

### Installation
```
git clone https://github.com/oooolga/Ctrl-V
cd Ctrl-V
python setup.py develop
```

# Data Directory
In the training and evaluation script, set the `DATASET_PATH` to the root of the dataset folder. Within this folder, you will find the extracted dataset subfolders. The dataset root folder should be organized in the following format: 	
```
Datasets/
├── bdd100k
├── kitti
└── vkitti_2.0.3
```

# Train a Bounding-box Predictor
A demo script is available at [demo_train_bbox_predict.sh](./scripts/train_scripts/demo_train_bbox_predict.sh).

To run [demo_train_bbox_predict.sh](./scripts/train_scripts/demo_train_bbox_predict.sh), set the `$DATASET_PATH` and `$OUT_DIR` to your desired path, and then execute 
```
bash ./scripts/train_scripts/demo_train_bbox_predict.sh
```

To resume training, set the `$NAME` variable to the name of the stopped experiment (e.g., `bdd100k_bbox_predict_240616_000000`). Ensure that you include `--resume_from_checkpoint latest` and that all the hyperparameter settings match those of the stopped experiment. After this setup, you can resume training by re-executing the training command.

To train on different sets, simply modify `DATASET` variable's value to `kitti`, `vkitti` or `bdd100k`. You can adjust the number of input frame conditions for your bounding-box predictor by changing the value of `num_cond_bbox_frames`. To change the last condition bounding-box frame to its trajectory frame, enable `if_last_frame_trajectory`.

# Train a Box2Video Model
A demo script is available at [demo_train_video_box2video.sh](./scripts/train_scripts/demo_train_video_box2video.sh).

To run [demo_train_video_box2video.sh](./scripts/train_scripts/demo_train_video_box2video.sh), set the `$DATASET_PATH` and `$OUT_DIR` to your desired path, and then execute 
```
bash ./scripts/train_scripts/demo_train_video_box2video.sh
```

To resume training, set the `$NAME` variable to the name of the stopped experiment (e.g., `bdd100k_ctrlv_240616_000000`). Ensure that you include `--resume_from_checkpoint latest` and that all the hyperparameter settings match those of the stopped experiment. After this setup, you can resume training by re-executing the training command.

To train on different sets, simply modify `DATASET` variable's value to `kitti`, `vkitti` or `bdd100k`.

# Generate and Evaluate Videos

## Running the whole generation pipeline (bounding-box predictor+box2video)
Demo scripts are available at [eval_scripts](./scripts/eval_scripts/).

To generate videos using the entire generation pipeline (predict bounding boxes and generate videos based on the predicted bounding box sequences), set the following variables in the `demo_eval_overall_{}.sh` scripts: `$DATASET_PATH`, `$OUT_DIR`, `$BOX2VIDEO_DIR`, and `$BBOX_MODEL_DIR`, and then execute 
```
bash ./scripts/eval_exripts/demo_eval_overall_{}.sh
```
For each input sample, the pipeline will predict five bounding-box sequences and select the one with the highest mask-IoU score to generate the final video. We evaluate bounding-box prediction metrics during the generation process, and the results are uploaded to the W&B dashboard.

The generated videos are also uploaded to the W&B dashboard. You can find a local copy of the generated videos in your W&B folder at `$OUT_DIR/wandb/run-{run_id}/files/media/videos`.

## Running the teacher-forced Box2Video generation pipeline
TODO

## Evaluations
### FVD, LPIPS, SSIM and PSNR
TODO

### YOLOv8 Detector and mAP Scores
TODO

# Credits

Our library is built on the work of many brilliant researchers and developers. We're grateful for their contributions, which have helped us with this project. Special thanks to the following repositories for providing valuable tools that have enhanced our project:


- @huggingface's [diffusion model library](https://github.com/huggingface/diffusers/).
- @ultralytics's [yolov8 library](https://github.com/ultralytics/ultralytics).

# Citation

```bibtex
@misc{luo2024ctrlv,
      title={Ctrl-V: Higher Fidelity Video Generation with Bounding-Box Controlled Object Motion}, 
      author={Ge Ya Luo and Zhi Hao Luo and Anthony Gosselin and Alexia Jolicoeur-Martineau and Christopher Pal},
      year={2024},
      eprint={2406.05630},
      archivePrefix={arXiv}
}
```