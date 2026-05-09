# Insights Report

## Experimental Setup

- Current project phase: Phase 5 pilot validation (long dose-8 A/B).
- Data split: Deterministic case-level split (`val_ratio=0.2`) on `data/processed/BraTS2023` using sorted case order (`split_seed=null` / no shuffle).
- Preprocessing: Load four modalities, channel-first, intensity normalization (`nonzero=True`, `channel_wise=True`), training random crop (`96^3`, smoke run with `num_samples=1`).
- Model: MONAI 3D UNet (`in_channels=4`, `out_channels=4`) trained for CPU smoke validation.
- Optimizer and schedule: Adam (`lr=1e-3`), no scheduler in smoke run.
- Evaluation metrics: Dice mean, HD95 mean, ECE.

## Quantitative Results

| Setting | Dice Mean | Dice ET | Dice TC | Dice WT | HD95 Mean | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline segmentation | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 |
| Baseline segmentation (CPU medium) | 0.0525 | 0.0158 | 0.1165 | 0.0250 | 94.1585 | 0.5455 |
| SegResNet (CPU medium quick-eval) | 0.0009 | 0.0000 | 0.0000 | 0.0028 | 175.0522 | 0.3150 |
| Baseline segmentation (CPU long v1) | 0.1237 | 0.0498 | 0.0374 | 0.2839 | 88.5451 | 0.3570 |
| Baseline + uncertainty analysis | 0.0891 | 0.0287 | 0.0155 | 0.0817 | 100.7852 | 0.0566 |
| Baseline + synthetic augmentation (label-preserving) | 0.0335 | 0.0147 | 0.0707 | 0.0150 | 121.3415 | 0.4941 |

## Qualitative Findings

- Case 1 (`BraTS-GLI-00000-000`): segmentation overlay, Grad-CAM, and uncertainty maps were generated successfully from CPU checkpoints.
- Case 2 (`BraTS-GLI-00001-000`): segmentation overlay, Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.
- Case 3 (`BraTS-GLI-00001-001`): segmentation overlay, Grad-CAM, SHAP, and uncertainty artifacts generated with CPU-bounded settings.

## Explainability Findings

- Grad-CAM: generated for 3 cases under `results/xai/smoke_v2/` with stable execution on padded full volumes.
- SHAP modality attribution: generated for 3 cases with CPU-bounded `nsamples=20`.

## Uncertainty Findings

- Boundary uncertainty: generated for 3 cases under `results/uncertainty/smoke_v2/` using MC Dropout (`passes=10`), suitable for visual boundary-risk inspection.
- Calibration behavior: global ECE in smoke setting is 0.0688; larger runs are needed for stable calibration claims.

## Synthetic Data Findings

- Sample realism: 8 synthetic panels were generated from the 2-epoch CPU VAE checkpoint (`results/generated_smoke_v2/`), with coherent multi-modal structure and expected blur for early-stage training.
- Performance impact: the untuned label-preserving synthetic recipe reduced Dice and worsened HD95 versus the baseline medium run, but the tuned recipe improved HD95 and nearly matched baseline Dice on a per-run basis.
- Multi-seed medium check: a 3-seed medium A/B with the tuned recipe (`seed=42,43,44`, 2 epochs, 20 train batches, 6 val batches) showed mean `delta_dice=-0.000117`, mean `delta_hd95=-2.442`, and mean `delta_ece=-0.000433`, indicating the tuned synthetic recipe is close to neutral and slightly better on boundary quality.
- Larger synthetic pool check: increasing tuned synthetic cases to 64 under the same 3-seed medium setup produced mean `delta_dice=-0.001782`, mean `delta_hd95=+6.082`, and mean `delta_ece=-0.013772`, indicating calibration improved slightly but segmentation quality regressed.
- Multi-seed quick check: a 3-seed quick A/B (`seed=42,43,44`) showed near-parity overall (`mean delta_dice=-0.000115`, `mean delta_hd95=-0.361`, `mean delta_ece=+0.0019`), indicating no reliable gain at current synthetic recipe strength.
- Unified trade-off view: consolidated ranking across baseline, untuned-16, tuned-16, and tuned-64 (`results/tables/variant_comparison_medium_multiseed.csv`) places tuned-16 first for balanced segmentation-quality/calibration trade-offs in this bounded setup.
- Dose sweep result: tuned-dose medium multiseed sweep (`dose=8,16,24,32`, `seed=42,43,44`) selected dose 8 as best in `results/tables/dose_sweep_medium_summary.csv`, with mean `delta_dice=+0.001589`, `delta_hd95=-1.927`, and near-flat calibration (`delta_ece=+0.000315`).
- Phase 5 pilot check: one-seed longer bounded A/B (`seed=42`, `epochs=3`, `case-limit=64`, `dose=8`) produced a small Dice gain but weaker boundary/calibration metrics (`delta_dice=+0.001751`, `delta_hd95=+3.103`, `delta_ece=+0.005041`) in `results/tables/seed_ablation_phase5_pilot_seed42_summary.csv`, reinforcing the need for the full 3-seed long campaign.

## Conclusions

- Best current setting: CPU long v1 baseline (`epochs=5`, `case-limit=64`) currently provides the strongest segmentation metrics in this repository.
- Main failure modes: dependency/import gaps, class-index mismatch in early metric code, and full-volume shape/memory issues (now mitigated via sliding-window inference + streaming ECE).
- Next three experiments: (1) complete the full 3-seed Phase 5 long A/B at tuned synthetic dose 8; (2) run longer CPU baselines (10+ epochs) with fixed random seeds; (3) run multi-seed XAI/uncertainty analysis over larger case subsets.
- Immediate implementation recommendation: keep tuned synthetic dose 8 as the active candidate augmentation setting, but do not promote it as final until the full 3-seed long campaign confirms the trend.

## GPU Scale-Up Notes

- Estimated acceleration for this pipeline with a modern single GPU is approximately 6x-20x for segmentation and 8x-25x for VAE training versus current CPU runs.
- This enables longer and more statistically stable studies in practical wall-clock time: deeper training schedules, ablations, and case-wise qualitative reviews.
- Recommended first GPU campaign: full-train baseline, followed by uncertainty calibration and synthetic augmentation A/B comparison under matched splits.

No GPU runs were executed in this project environment so far; all reported metrics and artifacts were generated on CPU only.
