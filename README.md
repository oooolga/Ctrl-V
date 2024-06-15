### This page is currently under construction

<p align="center">
<a href="https://oooolga.github.io/ctrl-v.github.io/">
<img src="./statics/logo.png" height="80" alt="logo">
</a>
</p>

# Ctrl-V: Higher Fidelity Video Generation with Bounding-Box Controlled Object Motion

<p align="left">
<a href="https://arxiv.org/abs/2406.05630" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2406.05630-b31b1b.svg?style=flat" /></a>
<a href="https://paperswithcode.com/paper/ctrl-v-higher-fidelity-video-generation-with">
    <img alt="Static Badge" src="https://img.shields.io/badge/paper_with_code-link-turquoise?logo=paperswithcode" />
</a>
<p align="center">
<img src="./statics/CtrlV_teaserv2.png" height="480" alt="teaser">
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

# Evaluate

# Credits

Our library is built on the work of many brilliant researchers and developers. We're grateful for their contributions, which have helped us create a better API. We want to thank the following projects for their significant contributions:


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
```