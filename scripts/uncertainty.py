from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Protobuf gencode version", category=UserWarning)

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, DivisiblePadd, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd

from models.segmentation import create_segmentation_model


def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo Dropout uncertainty for BraTS segmentation.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--case-dir", type=str, required=True)
    p.add_argument("--passes", type=int, default=20)
    p.add_argument("--spatial-size", type=int, default=128)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="results/uncertainty/uncertainty_map.png")
    p.add_argument("--quiet-warnings", action="store_true")
    return p.parse_args()


def load_case(case_dir: Path):
    stem = case_dir.name
    sample = {
        "image": [
            str(case_dir / f"{stem}_t1.nii.gz"),
            str(case_dir / f"{stem}_t1ce.nii.gz"),
            str(case_dir / f"{stem}_t2.nii.gz"),
            str(case_dir / f"{stem}_flair.nii.gz"),
        ]
    }
    tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            DivisiblePadd(keys=["image"], k=16),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
    out = tf(sample)
    return out["image"].unsqueeze(0)


def main():
    args = parse_args()
    if args.quiet_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = create_segmentation_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    enable_dropout(model)

    x = load_case(Path(args.case_dir)).to(device)
    roi_size = (args.spatial_size, args.spatial_size, args.spatial_size)
    sum_p = None
    sum_p2 = None
    with torch.no_grad():
        for _ in range(args.passes):
            logits = sliding_window_inference(
                x,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.overlap,
            )
            p = torch.softmax(logits, dim=1)
            if sum_p is None:
                sum_p = torch.zeros_like(p)
                sum_p2 = torch.zeros_like(p)
            sum_p += p
            sum_p2 += p * p

    n = max(1, int(args.passes))
    mean_p = sum_p / n
    mean_p2 = sum_p2 / n
    var = mean_p2 - mean_p * mean_p
    if n > 1:
        var = var * (n / (n - 1))
    var = torch.clamp(var, min=0.0)
    uncertainty = var.mean(dim=1).squeeze().cpu().numpy()
    flair = x[0, 3].detach().cpu().numpy()
    z = flair.shape[2] // 2
    flair_slice = flair[:, :, z]
    unc_slice = uncertainty[:, :, z]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(flair_slice, cmap="gray")
    plt.title("FLAIR slice")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(unc_slice, cmap="inferno")
    plt.title("MC Dropout Uncertainty")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved uncertainty map to {args.out}")


if __name__ == "__main__":
    main()
