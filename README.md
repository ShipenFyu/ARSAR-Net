# ARSAR-Net: Adaptively Regularized SAR Imaging Network with Efficient Unfolding

**This paper has been accepted by SCIENCE CHINA Information Sciences. [[PDF]](https://arxiv.org/pdf/2506.18324)**

**Fu S P, Chen Y F, Zhang Z, et al. Adaptively regularized SAR imaging network with efficient unfolding. Sci China Inf Sci,
2026, 69(8): 180306, https://doi.org/10.1007/s11432-025-5024-2**

## Abstract

Developed from sparse reconstruction approaches, deep unfolding networks (DUNs) have constituted an emerging method for synthetic aperture radar (SAR) imaging, offering fast convergence and data-driven learning. However, baseline unfolding networks, derived from iterative sparse reconstruction algorithms such as alternating direction method of multipliers (ADMM), lack generalization capability across scenes, as their regularizers are empirically designed and keep unchanged during imaging. In this study, we introduce a learnable regularizer to the unfolding network and propose an adaptively regularized SAR imaging network (ARSAR-Net) for scene-agnostic imaging (imaging across heterogeneous scenes of varying sparsity levels). In practice, the vanilla ARSAR-Net suffers from inherent structural limitations in 2D signal processing, primarily due to its reliance on matrix inversion. To conquer this, we further develop an ADMM without matrix inversion for efficient unfolding, by designing linear operations to replace the time-consuming matrix inversion operations. These advancements establish a new paradigm for efficient and scene-agnostic SAR imaging systems.

## Installation

The supplied Conda environment uses Python 3.10, PyTorch 2.3.1, and CUDA 11.8. An NVIDIA GPU is recommended for training and full-resolution inference.

```bash
git clone https://github.com/ShipenFyu/ARSAR-Net.git
cd ARSAR-Net
conda env create -f environment.yml
conda activate arsar-net
```

If your system requires another CUDA version or a CPU-only build, adjust the PyTorch entries in `environment.yml` before creating the environment.

## Data Preparation

Training and test data are stored as NumPy arrays containing complex-valued SAR images and echoes. Dataset directory names and `.npy` filenames are user-defined and are passed directly to the training or evaluation scripts.

```text
data/
|-- <dataset_name>/
|   |-- <train_image_file>.npy
|   |-- <train_echo_file>.npy
|   |-- <test_image_file>.npy
|   `-- <test_echo_file>.npy
`-- downsampling_method/
    |-- 50pct_method.npy
    |-- 75pct_method.npy
    `-- 100pct_method.npy
```

Each `.npy` file should contain a batch of 2D samples with NumPy dtype `complex64`. The current SAR configuration in `utils/config.py` uses 512 azimuth samples and 512 range samples. The value of `--down_rate` must match the sampling rate used to generate the echo data.

The `data/downsampling_method` directory name and sampling filenames are fixed because they are loaded by `utils/observation_matrix.py`. Sampling matrices for rates 0.5, 0.75, and 1.0 are included. To use another rate, generate the corresponding sampling file first.

MATLAB v7.3 arrays can be converted with:

```bash
python data/mat2np.py
```

The input paths, output paths, and MATLAB variable names used by this utility can be configured in `data/mat2np.py`.

## Training

Train ARSAR-Net Swift at 50% azimuth sampling:

```bash
python train.py \
  --train_image ./data/<dataset_name>/<train_image_file>.npy \
  --train_echo ./data/<dataset_name>/<train_echo_file>.npy \
  --weights_dir ./weights \
  --device cuda:0 \
  --network arsar \
  --regularization swift \
  --down_rate 0.5 \
  --layer_num 9 \
  --batch_size 4 \
  --epochs 80 \
  --lr 2e-5
```

For ARSAR-Net Pro, replace `--regularization swift` with `--regularization pro`. Checkpoints are saved under `<weights_dir>/<date>/`.

Resume training from a checkpoint with:

```bash
python train.py \
  --train_image ./data/<dataset_name>/<train_image_file>.npy \
  --train_echo ./data/<dataset_name>/<train_echo_file>.npy \
  --weights_dir ./weights \
  --network arsar \
  --regularization pro \
  --checkpoint ./weights/<date>/<checkpoint>.pt
```

## Evaluation

Evaluate a trained ARSAR-Net checkpoint:

```bash
python test.py \
  --test_image ./data/<dataset_name>/<test_image_file>.npy \
  --test_echo ./data/<dataset_name>/<test_echo_file>.npy \
  --weight ./weights/<date>/<checkpoint>.pt \
  --device cuda:0 \
  --network arsar \
  --regularization swift \
  --down_rate 0.5 \
  --layer_num 9 \
  --batch_size 4
```

The script reports NRMSE, PSNR, and SSIM, and saves reconstruction figures under `images/<regularization>/<sampling-rate>/`.

### Explicit-Regularization Baselines

Run the classical CSA, ISTA-L1, or ADMM-TV reconstruction baseline with `test_e.py`:

```bash
python test_e.py --test_image ./data/<dataset_name>/<test_image_file>.npy --test_echo ./data/<dataset_name>/<test_echo_file>.npy --device cuda:0 --regularization csa --down_rate 0.5
python test_e.py --test_image ./data/<dataset_name>/<test_image_file>.npy --test_echo ./data/<dataset_name>/<test_echo_file>.npy --device cuda:0 --regularization l1  --down_rate 0.5
python test_e.py --test_image ./data/<dataset_name>/<test_image_file>.npy --test_echo ./data/<dataset_name>/<test_echo_file>.npy --device cuda:0 --regularization tv  --down_rate 0.5
```

## Repository Structure

```text
ARSAR-Net/
|-- data/                       # Data conversion and sampling utilities
|-- logs/                       # Training logging utilities
|-- models/
|   |-- arsar_net.py            # ARSAR-Net Swift and Pro
|   |-- pnp_net.py              # Plug-and-play unfolding components
|   `-- solver.py               # Explicit reconstruction algorithms
|-- utils/
|   |-- config.py               # SAR system configuration
|   |-- evaluate.py             # NRMSE, PSNR, and SSIM
|   `-- observation_matrix.py   # Sampling operators
|-- train.py                    # Training entry point
|-- test.py                     # Network evaluation entry point
`-- test_e.py                   # Explicit-regularization evaluation
```

## Citation

If this work is useful in your research, please cite it:

```bibtex
@article{shipingadaptively,
  title={Adaptively regularized SAR imaging network with efficient unfolding},
  author={Shiping, Fu and Yufan, Chen and Zhe, Zhang and Qixiang, Ye},
  journal={SCIENCE CHINA Information Sciences},
  publisher={Science China Press}
}
```

## Acknowledgements

This work was particularly inspired by **ADMM-CSNet**, which demonstrated how the alternating direction method of multipliers can be unfolded into a trainable network for image compressive sensing. We gratefully acknowledge this important source of inspiration:

Yang, Y., Sun, J., Li, H., & Xu, Z. (2018). ADMM-CSNet: A deep learning approach for image compressive sensing. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *42*(3), 521-538.
