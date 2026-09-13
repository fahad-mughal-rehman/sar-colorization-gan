# SAR Image Colorization — Attention GAN

Translating grayscale Synthetic Aperture Radar (SAR) imagery into realistic optical (RGB) imagery using a conditional GAN with attention.

<!--
Once you have real results from a training run, replace this block with:
![results](assets/results/showcase.png)
-->
> 🖼️ Add your results image here — see [Adding your results](#adding-your-results) below.

## Overview

SAR sensors image the ground regardless of cloud cover or daylight, but the resulting grayscale imagery is hard to interpret visually. This project trains a conditional GAN to translate SAR imagery into optical-style RGB imagery, so the output is easier to read while keeping the availability advantages of radar.

**Architecture**
- **Generator:** U-Net with a self-attention block at the 16×16 bottleneck (SAGAN-style) and attention gates on every skip connection (Attention U-Net style, Oktay et al.)
- **Discriminator:** Multi-scale PatchGAN — three discriminators at full, half, and quarter resolution (Pix2PixHD style)
- **Losses:** LSGAN adversarial + feature matching + L1 + VGG19 perceptual + differentiable SSIM

**Training details**
- 256×256 paired SAR/optical images, deterministic train/val split (fixed seed) for reproducible evaluation
- Mixed-precision (AMP) training with automatic checkpoint resume, built for Kaggle's session time limits
- Evaluated with PSNR and SSIM on a held-out test split

## Dataset

[Paired SAR-Optical Dataset (16K Images)](https://www.kaggle.com/datasets/anjaliikakde/paired-sar-optical-dataset-16k-images) on Kaggle. Each image is a single PNG with the SAR tile on the left half and the paired optical tile on the right half.

## Repository structure

```
sar-colorization-gan/
├── notebooks/
│   └── sar_colorization_attention_gan.ipynb   # training + evaluation, run top to bottom
├── scripts/
│   └── crop_results.py                        # turns a saved comparison grid into labeled showcase images
├── assets/
│   └── results/                                # put your own output images here (see below)
├── requirements.txt
└── README.md
```

## Getting started

1. Open `notebooks/sar_colorization_attention_gan.ipynb` on a GPU environment (developed for Kaggle P100/T4; a local CUDA GPU works too).
2. Download the dataset above and point `CFG["DATA_DIR"]` at it.
3. Run the notebook top to bottom. Training checkpoints and sample grids are written to `checkpoints/` and `samples/`; re-running the notebook resumes automatically from the last checkpoint.
4. The evaluation section at the bottom loads a checkpoint and reports PSNR/SSIM on the held-out split, plus saves comparison grids (SAR | Generated | Ground Truth) to `test_results/`.

### Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/sar_colorization_attention_gan.ipynb
```

## Results

| Metric | Value |
|---|---|
| PSNR | _fill in from your `test_results/metrics.txt`_ |
| SSIM | _fill in from your `test_results/metrics.txt`_ |
| Test set size | _fill in_ |

### Adding your results

The notebook's evaluation section saves comparison grids like `test_results/comparison_000.png`, each one a single image stacked as **row 1 = SAR input, row 2 = generated, row 3 = ground truth**, with one column per sample in the batch.

To turn those into clean, labeled images for this README or a portfolio/LinkedIn post:

```bash
python scripts/crop_results.py \
    --grid path/to/comparison_000.png \
    --img-size 256 --padding 2 \
    --columns 0 1 2 3 \
    --out assets/results
```

This saves one labeled SAR/Generated/Ground-Truth panel per column plus a combined `showcase.png` strip. Point the README image at the top of this file to `assets/results/showcase.png` once it exists.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Attention U-Net: Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*
- Multi-scale PatchGAN + feature matching: Wang et al., *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* (Pix2PixHD)
- Self-attention block: Zhang et al., *Self-Attention Generative Adversarial Networks* (SAGAN)
