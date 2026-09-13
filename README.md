
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
| PSNR | 17.669 ± 2.929 dB |
| SSIM | 0.3393 ± 0.1155 |
| Test set size | 319 images |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Attention U-Net: Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*
- Multi-scale PatchGAN + feature matching: Wang et al., *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* (Pix2PixHD)
- Self-attention block: Zhang et al., *Self-Attention Generative Adversarial Networks* (SAGAN)