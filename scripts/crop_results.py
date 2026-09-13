"""
crop_results.py

Turns one of the comparison grids saved by the notebook
(samples/epoch_XXX.png or test_results/comparison_XXX.png) into clean,
labeled showcase images for a README, portfolio, or social post.

Those grids are produced by the notebook as:
    grid_tensor = torch.cat([sar_rgb, fake, ground_truth], dim=0)
    vutils.make_grid(grid_tensor, nrow=batch_size)   # default padding=2

which lays out exactly 3 rows (SAR / Generated / Ground Truth) with
`batch_size` columns, each cell `img_size x img_size` plus `padding` pixels
of border on every side.

Usage:
    python crop_results.py --grid samples/epoch_060.png \
        --img-size 256 --padding 2 \
        --columns 0 2 5 \
        --out assets/results

Adjust --img-size / --padding if you changed CFG["IMG_SIZE"] or called
make_grid with a non-default padding in your own copy of the notebook.
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

ROW_LABELS = ["SAR Input", "Generated (Colorized)", "Ground Truth"]


def cell_box(row, col, img_size, padding):
    """Pixel box (left, top, right, bottom) of one grid cell.

    torchvision.utils.make_grid draws `padding` pixels of border around every
    cell, including before the first row/column, so cell i starts at
    i * (img_size + padding) + padding.
    """
    step = img_size + padding
    left = col * step + padding
    top = row * step + padding
    return (left, top, left + img_size, top + img_size)


def label_panel(img, text, font_size=18):
    """Return a copy of `img` with a title bar reading `text` above it."""
    bar_h = font_size + 14
    panel = Image.new("RGB", (img.width, img.height + bar_h), "white")
    panel.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((panel.width - tw) // 2, 7), text, fill="black", font=font)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="Path to a saved comparison grid PNG")
    ap.add_argument("--img-size", type=int, default=256, help="CFG['IMG_SIZE'] used in the notebook")
    ap.add_argument("--padding", type=int, default=2, help="make_grid padding (2 is the torchvision default)")
    ap.add_argument("--columns", type=int, nargs="+", default=None,
                     help="Which sample columns to extract (0-indexed). Default: all columns.")
    ap.add_argument("--out", default="assets/results", help="Output directory")
    ap.add_argument("--strip-only", action="store_true",
                     help="Only save the combined showcase strip, skip individual panels")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = Image.open(args.grid).convert("RGB")

    step = args.img_size + args.padding
    n_cols = (grid.width - args.padding) // step
    n_rows = (grid.height - args.padding) // step
    if n_rows != 3:
        print(f"Warning: expected 3 rows (SAR/Generated/GT), found {n_rows} — "
              "check --img-size/--padding match how the grid was made.")

    columns = args.columns if args.columns is not None else list(range(n_cols))
    columns = [c for c in columns if 0 <= c < n_cols]
    if not columns:
        raise SystemExit(f"No valid columns to extract (grid has {n_cols} columns).")

    showcase_columns = []
    for col in columns:
        row_imgs = []
        for row in range(min(n_rows, 3)):
            crop = grid.crop(cell_box(row, col, args.img_size, args.padding))
            labeled = label_panel(crop, ROW_LABELS[row])
            row_imgs.append(labeled)

        # Stack this column's 3 labeled rows side by side into one panel.
        gap = 10
        w = sum(im.width for im in row_imgs) + gap * (len(row_imgs) - 1)
        h = max(im.height for im in row_imgs)
        panel = Image.new("RGB", (w, h), "white")
        x = 0
        for im in row_imgs:
            panel.paste(im, (x, 0))
            x += im.width + gap

        if not args.strip_only:
            out_path = os.path.join(args.out, f"sample_{col:02d}.png")
            panel.save(out_path)
            print("Saved", out_path)
        showcase_columns.append(panel)

    # Combined vertical showcase strip (good for a README hero image / LinkedIn post).
    gap = 16
    sw = max(p.width for p in showcase_columns)
    sh = sum(p.height for p in showcase_columns) + gap * (len(showcase_columns) - 1)
    showcase = Image.new("RGB", (sw, sh), "white")
    y = 0
    for p in showcase_columns:
        showcase.paste(p, ((sw - p.width) // 2, y))
        y += p.height + gap
    showcase_path = os.path.join(args.out, "showcase.png")
    showcase.save(showcase_path)
    print("Saved", showcase_path)


if __name__ == "__main__":
    main()
