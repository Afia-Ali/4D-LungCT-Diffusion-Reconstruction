# Phase-Conditioned Diffusion Models for 4D Lung CT Reconstruction

This repository implements an end-to-end medical image processing and generative modeling pipeline designed to synthesize continuous respiratory states from 4D-CT lung scans. By utilizing a phase-conditioned 3D Diffusion U-Net with Feature-wise Linear Modulation (FiLM), this project aims to improve tumor tracking accuracy during radiation therapy.

## Project Overview
Accurate tumor tracking during respiration is critical for safe and precise radiotherapy delivery[cite: 28, 29]. This project builds a deep learning workflow that takes a primary inhalation anchor scan ($T00$), couples it with target phase identifiers, and uses a generative diffusion network to reconstruct intermediate respiratory phases ($T10$–$T90$)[cite: 32, 44].

## System Architecture & Pipeline
The project is divided into distinct, modular stages:

1. **Data Engineering & Preprocessing:** Handles raw binary files from the DIR-Lab dataset[cite: 31, 32]. Standardizes mismatched resolutions (resizing Cases 6–10 from $512 \times 512$ down to $256 \times 256$ to match Cases 1–5) and resamples all volumes to a uniform 3D tensor shape of `[1, 64, 256, 256]`[cite: 35]. Intensities are normalized to a stable lung window of `[0.0, 1.0]` optimized for diffusion models[cite: 36].
2. **Registration-Based Augmentation:** Implements ANTsPy's Symmetric Normalization (SyN) algorithm to compute deformable warps between extreme breathing phases ($T00$ and $T50$)[cite: 42]. This interpolates physics-aligned intermediate steps to supplement training data[cite: 42, 43].
3. **3D Phase-Locked Spatial Augmentation:** Multiplies the baseline clean volumes by applying synchronized 3D spatial transformations (rotations up to $10^\circ$, scaling, flips, and translations) across all 10 respiratory phases simultaneously to scale the training pool to **801 total 3D volumes** without breaking anatomical alignment.
4. **Phase-Conditioned 3D Diffusion U-Net:** A generative model that uses a forward noise scheduler and reverses it via a 3D U-Net backbone[cite: 17, 44]. Conditioning information (phase identifiers and noise levels) is embedded across network layers using Feature-wise Linear Modulation (FiLM)[cite: 44, 45].
5. **Clinical Evaluation:** Quantifies generation accuracy by calculating the Target Registration Error (TRE) in millimeters against expert-annotated ground-truth anatomical landmarks on held-out validation cases[cite: 52, 53, 54].

## Repository Structure
```text
├── data/                  # Local directory for raw/processed data paths
├── src/
│   ├── preprocessing.py   # Raw data parsing, resampling, and normalization
│   ├── augmentation.py    # ANTsPy SyN deformation field interpolation & TorchIO loops
│   ├── dataset.py         # Custom PyTorch Dataset & DataLoader
│   ├── models/
│   │   └── unet3d.py      # 3D U-Net backbone with FiLM conditioning
│   ├── diffusion.py       # Forward noise scheduler and sampling loop
│   └── evaluate.py        # TRE landmark distance evaluation metrics
├── notebooks/
│   └── Pipeline_Rough.ipynb
└── README.md

```

## Getting Started

### Prerequisites

* Python 3.10+
* PyTorch (with CUDA support)
* TorchIO
* SimpleITK
* ANTsPy

### Installation

```bash
git clone [https://github.com/your-username/Phase-Conditioned-4DCT-Diffusion.git](https://github.com/your-username/Phase-Conditioned-4DCT-Diffusion.git)
cd Phase-Conditioned-4DCT-Diffusion
pip install -r requirements.txt

```

## Dataset Split & Hygiene

To avoid data leakage, dataset splitting is strictly managed by patient identity rather than individual phase frames:

* 
**Training Matrix:** 8 clinical cases (Cases 1, 2, 4, 5, 6, 7, 8, 10) subjected to phase-locked spatial and registration augmentations.


* 
**Validation/Testing Matrix:** Cases 3 and 9 are held out completely. Their expert-annotated landmarks serve as the absolute ground truth for final TRE metric validation and are never seen during training.



## Clinical Outcome

* 
**Primary Metric:** Target Registration Error (TRE) measured in millimeters ($mm$).


* 
**Objective:** Comparing our 3D diffusion network against a baseline 2D U-Net. Achieving a lower TRE translates directly into sharper tumor boundary tracking during real-time breathing cycles, enabling safer, high-precision radiation dosing to target tissues while protecting surrounding healthy organs.



```

```