# Execution Runbook

This runbook helps generate real evidence quickly and consistently.

## Step 1: Set up environment

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Confirm GPU and package versions.
4. On Windows, run the commands from the same interpreter you installed into, or invoke the explicit venv Python path directly, so campaign subprocesses do not fall back to a different system/conda interpreter.

## Step 2: Prepare data

1. Download BraTS 2023 using one of the supported options below.
2. Use the dataset preparation script to normalize file layout.
3. Verify each prepared case has T1, T1ce, T2, FLAIR, and segmentation mask.

Official source note:
- The BraTS 2023 challenge data is officially hosted on Synapse.
- Kaggle and Hugging Face options are practical mirrors/community-hosted copies and should be validated against expected naming and modalities.

Option A: Kaggle API
- Configure Kaggle credentials (`~/.kaggle/kaggle.json` or environment variables).
- Download and prepare:
	- `python scripts/download_brats.py --source kaggle --kaggle-dataset <owner/dataset-slug> --extract-zips`
	- or `python scripts/download_brats.py --source kaggle --kaggle-competition <competition-slug> --extract-zips`

Option B: Hugging Face Hub
- Authenticate if required (`huggingface-cli login`).
- Download and prepare:
	- `python scripts/download_brats.py --source huggingface --hf-repo-id <dataset-repo-id> --extract-zips`

If data has already been downloaded manually:
- Put raw files under `data/raw/BraTS2023/` and run:
	- `python scripts/download_brats.py --source none --extract-zips`

Recommended for Synapse zip downloads in this repository layout:
- Place archives under `data/raw/BraTS2023/archives/GLI`.
- Prepare GLI cases for segmentation with:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023 --extract-zips --link-mode hardlink --include-case-prefix BraTS-GLI`

Optional challenge-specific preparation commands:
- MEN:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023-MEN --extract-zips --link-mode hardlink --include-case-prefix BraTS-MEN`
- PED:
	- `python scripts/download_brats.py --source none --raw-dir data/raw/BraTS2023/archives --processed-dir data/processed/BraTS2023-PED --extract-zips --link-mode hardlink --include-case-prefix BraTS-PED`

Quick verification command:
- `python scripts/download_brats.py --source none --dry-run`

## Step 3: Baseline segmentation

CPU smoke-test command (recommended before full training):
- `python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --quick-cpu --device cpu --out checkpoints/best_model_cpu_smoke.pt --quiet-warnings`

Run:
- `python scripts/train_segmentation.py --data-dir data/processed/BraTS2023 --model unet --epochs 50`

Then evaluate:
- `python scripts/evaluate.py --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model.pt --out results/metrics/baseline_metrics.json --quiet-warnings`

Note: `scripts/evaluate.py` computes ECE in streaming mode (no per-voxel list accumulation), so large volumes won't OOM RAM.

## Step 4: XAI and uncertainty

Run XAI:
- `python scripts/run_xai.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID> --out-dir results/figures`

Run uncertainty:
- `python scripts/uncertainty.py --checkpoint checkpoints/best_model.pt --case-dir data/processed/BraTS2023/<CASE_ID> --out results/figures/uncertainty_case_<CASE_ID>.png`

## Step 5: Generative extension

Train VAE:
- `python scripts/train_vae.py --data-dir data/processed/BraTS2023 --epochs 30 --out checkpoints/vae.pt`

Generate samples:
- `python scripts/generate_samples.py --checkpoint checkpoints/vae.pt --n 8 --out-dir results/figures`

## Step 5b: Label-preserving synthetic augmentation (trainable)

Generate labeled synthetic cases:
- `python -m scripts.generate_label_preserving_synthetic --source-dir data/processed/BraTS2023 --out-dir data/processed/BraTS2023_SYN --num-synthetic 16 --base-case-limit 32 --seed 42`

Train segmentation with synthetic train-only additions:
- `python -m scripts.train_segmentation --data-dir data/processed/BraTS2023 --train-extra-dir data/processed/BraTS2023_SYN --device cpu --epochs 2 --num-workers 0 --case-limit 32 --spatial-size 96 --num-samples 1 --max-train-batches 20 --max-val-batches 6 --seed 42 --split-seed -1 --out checkpoints/best_model_cpu_mid_plus_synth.pt --quiet-warnings`

Evaluate augmented checkpoint:
- `python -m scripts.evaluate --data-dir data/processed/BraTS2023 --checkpoint checkpoints/best_model_cpu_mid_plus_synth.pt --device cpu --out results/metrics/baseline_metrics_cpu_mid_plus_synth.json --num-workers 0 --case-limit 32 --max-val-batches 6 --spatial-size 96 --seed 42 --split-seed -1 --val-ratio 0.2 --quiet-warnings`

## Step 6: Fill logs and reports

1. Copy `docs/experiment_log_template.md` to `docs/experiment_log.md` and fill all fields.
2. Copy `docs/insights_report_template.md` to `docs/insights_report.md` and add findings.
3. Update `README.md` Results section with final numbers and links to figures.

## Step 6b: Phase 5 long multiseed campaign (recommended next)

Before launching these commands, make sure the active Python interpreter can import `monai` and the rest of `requirements.txt`.

Quick pilot (1 seed, bounded runtime) to validate the full loop:
- `python -m scripts.run_phase5_long_dose8_campaign --seeds 42 --epochs 3 --max-train-batches 20 --max-val-batches 6 --tag phase5_pilot_seed42 --skip-existing --quiet-warnings`

Full long campaign (3 seeds, tuned synthetic dose 8):
- `python -m scripts.run_phase5_long_dose8_campaign --seeds 42,43,44 --epochs 10 --max-train-batches 40 --max-val-batches 10 --dose 8 --tag phase5_long_dose8 --skip-existing --quiet-warnings`

If the tuned synthetic pool is missing, auto-generate it inline:
- `python -m scripts.run_phase5_long_dose8_campaign --seeds 42,43,44 --epochs 10 --max-train-batches 40 --max-val-batches 10 --dose 8 --tag phase5_long_dose8 --auto-generate-synthetic --skip-existing --quiet-warnings`

Expected key outputs:
- `results/tables/seed_ablation_phase5_long_dose8_summary.csv`
- `results/metrics/seed_42_baseline_phase5_long_dose8.json`
- `results/metrics/seed_42_plus_synth_dose8_phase5_long_dose8.json`
- `checkpoints/best_model_cpu_phase5_long_dose8_seed42_baseline.pt`
- `checkpoints/best_model_cpu_phase5_long_dose8_seed42_plus_synth_dose8.pt`

## Step 7: Final quality check

- Verify every metric cited in README has a corresponding file in `results/`.
- Verify every figure in README points to a generated output.
- Keep one clear ablation table and one concise conclusion section.
