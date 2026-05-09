from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    denom = cam.max() + 1e-8
    return cam / denom


def _overlay(slice_img: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(cam, cmap="jet", alpha=alpha)
    plt.axis("off")


def run_gradcam(
    model: torch.nn.Module,
    volume: torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: int = 1,
    out_file: str = "results/gradcam_overlay.png",
):
    """Compute a Grad-CAM map on a middle axial slice for qualitative explainability."""
    model.eval()
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients["value"] = grad_output[0].detach()

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    volume = volume.requires_grad_(True)
    logits = model(volume)
    score = logits[:, class_idx, ...].mean()

    model.zero_grad(set_to_none=True)
    score.backward()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=volume.shape[-3:], mode="trilinear", align_corners=False)

    cam_np = cam.squeeze().cpu().numpy()
    cam_np = _normalize_cam(cam_np)

    vol_np = volume.detach().squeeze().cpu().numpy()
    # Use FLAIR channel for visualization by default.
    flair = vol_np[3] if vol_np.ndim == 4 and vol_np.shape[0] >= 4 else vol_np[0]
    if flair.ndim == 3 and cam_np.ndim == 3:
        z = flair.shape[2] // 2
        _overlay(flair[:, :, z], cam_np[:, :, z])
    else:
        _overlay(flair, cam_np)
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    h1.remove()
    h2.remove()


def preprocess_single_case(modality_arrays: Iterable[np.ndarray]) -> torch.Tensor:
    transform = Compose([EnsureChannelFirst(channel_dim="no_channel"), NormalizeIntensity(nonzero=True)])
    stacked = np.stack(list(modality_arrays), axis=0)
    tensor = torch.as_tensor(stacked, dtype=torch.float32)
    tensor = transform(tensor)
    return tensor.unsqueeze(0)
