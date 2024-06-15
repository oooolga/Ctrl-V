# This page is currently under construction

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

# Datasets
In the training and evaluation script, set the `DATASET_PATH` to the root of the dataset folder. Within this folder, you will find the extracted dataset subfolders. The dataset root folder should be organized in the following format: 	
```
Datasets/
├── bdd100k
├── kitti
└── vkitti_2.0.3
```