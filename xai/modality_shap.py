from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _predict_from_modalities(model: torch.nn.Module, samples: np.ndarray, base_volume: torch.Tensor) -> np.ndarray:
    """samples shape: (N, 4), each value scales one MRI modality globally."""
    preds = []
    with torch.no_grad():
        for row in samples:
            x = base_volume.clone()
            for c in range(min(4, x.shape[1])):
                x[:, c] *= float(row[c])
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            # Use ET average probability as scalar target (BraTS ET label index = 3).
            et_idx = min(3, probs.shape[1] - 1)
            preds.append(float(probs[:, et_idx, ...].mean().item()))
    return np.asarray(preds, dtype=np.float32)


def run_modality_shap(
    model: torch.nn.Module,
    base_volume: torch.Tensor,
    out_file: str = "results/shap_modalities.png",
    nsamples: int = 100,
):
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install requirements.txt to enable modality attribution.") from exc

    model.eval()
    background = np.ones((16, 4), dtype=np.float32)
    samples = np.asarray(
        [
            [0.8, 1.0, 1.0, 1.0],
            [1.0, 0.8, 1.0, 1.0],
            [1.0, 1.0, 0.8, 1.0],
            [1.0, 1.0, 1.0, 0.8],
            [1.2, 1.0, 1.0, 1.0],
            [1.0, 1.2, 1.0, 1.0],
            [1.0, 1.0, 1.2, 1.0],
            [1.0, 1.0, 1.0, 1.2],
        ],
        dtype=np.float32,
    )

    explainer = shap.KernelExplainer(lambda z: _predict_from_modalities(model, z, base_volume), background)
    shap_values = explainer.shap_values(samples, nsamples=nsamples)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    labels = ["T1", "T1ce", "T2", "FLAIR"]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, mean_abs, color=["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"])
    plt.title("Modality Attribution via SHAP")
    plt.ylabel("Mean |SHAP value|")
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
