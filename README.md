# SDN
Stain Disentanglement Network for staining transformation in WSI.

# Requirement

```
Python 3.7
PyTorch 1.12.0
```

# Dataset

A HE stained dataset [MITOS-ATYPIA-14](https://mitos-atypia-14.grand-challenge.org/) provides frames scanned by two scanners: Aperio Scanscope XT and Hamamatsu Nanozoomer 2.0-HT. We reorganized the file structure as:

```
dataset/
├── testing
│   ├── A06_00Aa.tiff
│   └── H06_00Aa.tiff
│   └── ...
└── training
    ├── A03_00Aa.tiff
    └── H03_00Aa.tiff
    └── ...
```

# Training

In `train.py`, the process of training SDN ia shown. The default setting is to train SDN on frames whose names are started with H (Hamamatsu Nanozoomer 2.0-HT). During training, the program saves some validation results in `./output`.

# Testing

In `test.py`, we show the testing process using a reference image for random cropped patch transformation. With the aligned frames scanned by two scanners in MITOS-ATYPIA-14, SSIM and PSNR can be evaluated during testing.