from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, DivisiblePadd, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd

from models.segmentation import create_segmentation_model


def parse_args():
    p = argparse.ArgumentParser(description="Generate segmentation overlay figure for one BraTS case.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model_cpu_long_v1.pt")
    p.add_argument("--case-dir", type=str, required=True)
    p.add_argument("--spatial-size", type=int, default=128)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def load_case(case_dir: Path):
    stem = case_dir.name
    sample = {
        "image": [
            str(case_dir / f"{stem}_t1.nii.gz"),
            str(case_dir / f"{stem}_t1ce.nii.gz"),
            str(case_dir / f"{stem}_t2.nii.gz"),
            str(case_dir / f"{stem}_flair.nii.gz"),
        ],
        "label": str(case_dir / f"{stem}_seg.nii.gz"),
    }
    tf = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            DivisiblePadd(keys=["image", "label"], k=16),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    out = tf(sample)
    image = out["image"].unsqueeze(0)
    label = out["label"]
    return image, label


def plot_overlay(flair: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_file: Path):
    z = flair.shape[2] // 2
    flair_slice = flair[:, :, z]
    gt_slice = gt[:, :, z]
    pred_slice = pred[:, :, z]

    gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
    pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))

    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title("FLAIR", pad=12)
    axes[0].axis("off")

    axes[1].imshow(flair_slice, cmap="gray")
    axes[1].imshow(gt_mask, cmap="viridis", alpha=0.55, vmin=1, vmax=3)
    axes[1].set_title("Ground Truth Overlay", pad=12)
    axes[1].axis("off")

    axes[2].imshow(flair_slice, cmap="gray")
    axes[2].imshow(pred_mask, cmap="plasma", alpha=0.55, vmin=1, vmax=3)
    axes[2].set_title("Prediction Overlay", pad=12)
    axes[2].axis("off")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.90, wspace=0.06)
    fig.savefig(out_file, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = create_segmentation_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    image, label = load_case(Path(args.case_dir))
    image = image.to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=(args.spatial_size, args.spatial_size, args.spatial_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int16)

    flair = image[0, 3].detach().cpu().numpy()
    gt = label.squeeze(0).cpu().numpy().astype(np.int16)

    out_file = Path(args.out)
    plot_overlay(flair, gt, pred, out_file)
    print(f"Saved segmentation overlay to {out_file}")


if __name__ == "__main__":
    main()
