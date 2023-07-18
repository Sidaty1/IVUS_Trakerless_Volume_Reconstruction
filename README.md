# Trackerless Volume Reconstruction from Intraoperative Ultrasound Images

Official implementation of the MICCAI 2023 paper - [_Trackerless Volume Reconstruction from Intraoperative Ultrasound Images_](https://inria.hal.science/hal-04108248/file/MICCAI2023_Trakerless_Volume_Reconstruction_from_Intraoperative_Ultrasound_images.pdf)


# Introduction

We propose a method for trackerless ultrasound volume reconstruction in the context of minimally invasive surgery. It is based on a Siamese architecture, including a recurrent neural network that leverages the ultrasound image features and the optical flow to estimate the relative position of frames

![alt text](data/imgs/architecture.png "The input sequence is split into two equal sequences with a common frame. Both are used to compute a sparse optical flow. Gaussian heatmaps tracking M points are then combined with the first and last frame of each sequence to form the network’s input. We use a Siamese architecture based on Sequence to Vector (Seq2Vec) network. The learning is done by minimising the mean square error between the output and ground truth transformations.")


# Installation

```bash
pip install -r requirements.txt
```

#  Citations
```
@inproceedings{El_Hadramy_Verde2023unified,
    title ={Trackerless Volume Reconstruction from Intraoperative Ultrasound Images},
    author={El Hadramy, Sidaty and Verde, Juan and Beaudet Karl-Philippe and Padoy, Nicolas and Cotin, Stéphane},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year ={2023},
    organization={Springer}
}
```