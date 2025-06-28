# [Spatial-Coordinate Attention Network for Continuous Sign Language Recognition](https://github.com/tinh2044/SCAttenNet)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Introduction
We propose SCAttenNet, a novel spatial-coordinate attention network for continuous sign language recognition that leverages coordinate-aware processing to capture the intricate spatial-temporal dependencies inherent in sign language gestures. Our approach employs a dual-coordinate encoder-decoder framework with cross-attention mechanisms to process human pose keypoints effectively. The resulting framework separates X and Y coordinate processing to enhance spatial relationship modeling and achieves superior performance through innovative coordinate-aware processing and multi-scale feature extraction.

<img src="images/architecture.png" width="800">

## Performance

| Dataset | WER | Model | Training |
| :---: | :---: | :---: | :---: | 
| Phoenix-2014 | 18.1% | [ckpt](placeholder) | [config](configs/phoenix-2014_s2g.yaml) |
| Phoenix-2014T | 17.67% | [ckpt](placeholder) | [config](configs/phoenix-2014t_s2g.yaml) |
 

## Installation
```
conda create -n scattennet python==3.8
conda activate scattennet
# Please install PyTorch according to your CUDA version.
pip install -r requirements.txt
```

### Download

**Datasets**

Download datasets from their websites and place them under the corresponding directories in data/
* [Phoenix-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
* [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

**Pretrained Models**

We provide pretrained models [Phoenix-2014](placeholder) and [Phoenix-2014T](placeholder). Download this directory and place them under *pretrained_models*.

**Keypoints**
We provide human keypoints for two datasets, [Phoenix-2014](placeholder) and [Phoenix-2014T](placeholder), pre-extracted by MediaPipe/OpenPose. Please download them and place them under *data/Phoenix-2014t(Phoenix-2014)*.

## Training
```
# Train on Phoenix-2014-T dataset
python main.py --cfg_path configs/phoenix-2014t.yaml \
               --batch-size 8 \
               --epochs 100 \
               --device cuda

# Resume training from checkpoint
python main.py --cfg_path configs/phoenix-2014t.yaml \
               --resume outputs/SLR/Phoenix-2014-T/best_checkpoint.pth
```

## Evaluation
```
# Evaluate trained model
python main.py --cfg_path configs/phoenix-2014t.yaml \
               --eval \
               --resume outputs/SLR/Phoenix-2014-T/best_checkpoint.pth
```

<!-- ## Citations
```
@misc{SCAttenNet2024,
title = {SCAttenNet: Spatial-Coordinate Attention Network for Continuous Sign Language Recognition},
author = {Tinh Nguyen},
year = {2024},
url = {https://github.com/tinh2044/SCAttenNet},
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Phoenix-2014-T dataset providers
- PyTorch team for the deep learning framework
- Sign language research community

---
