from __future__ import annotations

import argparse
from pathlib import Path

import torch
from monai.transforms import Compose, DivisiblePadd, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd
from monai.transforms import ResizeWithPadOrCropd

from models.segmentation import create_segmentation_model
from xai.gradcam import run_gradcam
from xai.modality_shap import run_modality_shap


def parse_args():
    p = argparse.ArgumentParser(description="Run Grad-CAM and modality-level SHAP analysis.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--case-dir", type=str, required=True)
    p.add_argument(
        "--spatial-size",
        type=int,
        default=128,
        help="Center crop/pad size for XAI (keeps Grad-CAM/SHAP memory bounded).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--shap-nsamples", type=int, default=100)
    return p.parse_args()


def load_case(case_dir: Path, spatial_size: int):
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
            ResizeWithPadOrCropd(keys=["image"], spatial_size=(spatial_size, spatial_size, spatial_size)),
            DivisiblePadd(keys=["image"], k=16),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
    out = tf(sample)
    return out["image"].unsqueeze(0)


def pick_target_layer(model: torch.nn.Module):
    # For MONAI UNet, use the deepest block before decoding as Grad-CAM target.
    if hasattr(model, "model") and hasattr(model.model, "downsamples"):
        return model.model.downsamples[-1]
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv3d):
            return m
    raise RuntimeError("Could not infer a suitable target layer for Grad-CAM")


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = create_segmentation_model(ckpt.get("model_name", "unet")).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    x = load_case(Path(args.case_dir), spatial_size=args.spatial_size).to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_layer = pick_target_layer(model)
    run_gradcam(model, x, target_layer, out_file=str(out_dir / "gradcam_overlay.png"))

    if not args.skip_shap:
        run_modality_shap(
            model,
            x,
            out_file=str(out_dir / "shap_modalities.png"),
            nsamples=args.shap_nsamples,
        )

    print(f"Saved XAI outputs to {out_dir}")


if __name__ == "__main__":
    main()
