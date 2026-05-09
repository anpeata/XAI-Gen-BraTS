from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Phase 5 long multiseed A/B validation: baseline vs tuned synthetic dose 8 "
            "(or a user-specified dose)."
        )
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--data-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--synthetic-dir", type=str, default="data/processed/BraTS2023_SYN_tuned64")
    p.add_argument("--dose", type=int, default=8)
    p.add_argument("--seeds", type=str, default="42,43,44")

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--case-limit", type=int, default=64)
    p.add_argument("--spatial-size", type=int, default=96)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--max-train-batches", type=int, default=40)
    p.add_argument("--max-val-batches", type=int, default=10)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=-1)

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--metrics-dir", type=str, default="results/metrics")
    p.add_argument("--tables-dir", type=str, default="results/tables")
    p.add_argument("--tag", type=str, default="phase5_long_dose8")

    p.add_argument(
        "--auto-generate-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--synthetic-pool-size", type=int, default=64)
    p.add_argument("--synthetic-base-case-limit", type=int, default=32)
    p.add_argument("--synthetic-seed", type=int, default=142)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--quiet-warnings", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("No seeds provided")
    return seeds


def _printable_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(f"[RUN] {_printable_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_step(
    *,
    name: str,
    cmd: list[str],
    expected_outputs: list[Path],
    skip_existing: bool,
    dry_run: bool,
) -> None:
    def _is_ready_output(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_dir():
            return any(path.iterdir())
        return True

    if skip_existing and expected_outputs and all(_is_ready_output(p) for p in expected_outputs):
        outputs = ", ".join(str(p) for p in expected_outputs)
        print(f"[SKIP] {name} (outputs already exist: {outputs})")
        return
    print(f"\n=== {name} ===")
    run_cmd(cmd, dry_run=dry_run)


def maybe_generate_synthetic_pool(args: argparse.Namespace) -> None:
    synthetic_dir = Path(args.synthetic_dir)
    has_any_content = synthetic_dir.exists() and any(synthetic_dir.iterdir())
    if has_any_content:
        print(f"Using existing synthetic directory: {synthetic_dir}")
        return

    if not args.auto_generate_synthetic:
        raise FileNotFoundError(
            "Synthetic directory is missing or empty, and auto-generation is disabled. "
            "Provide a populated --synthetic-dir or omit --no-auto-generate-synthetic."
        )

    gen_cmd = [
        args.python,
        "-m",
        "scripts.generate_label_preserving_synthetic",
        "--source-dir",
        args.data_dir,
        "--out-dir",
        args.synthetic_dir,
        "--num-synthetic",
        str(args.synthetic_pool_size),
        "--base-case-limit",
        str(args.synthetic_base_case_limit),
        "--min-tumor-voxels",
        "1000",
        "--flip-prob",
        "0.25",
        "--scale-min",
        "0.97",
        "--scale-max",
        "1.03",
        "--shift-min",
        "-0.02",
        "--shift-max",
        "0.02",
        "--gamma-min",
        "0.97",
        "--gamma-max",
        "1.03",
        "--seed",
        str(args.synthetic_seed),
    ]
    run_step(
        name="Generate tuned synthetic pool",
        cmd=gen_cmd,
        expected_outputs=[synthetic_dir],
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


def build_train_cmd(
    *,
    args: argparse.Namespace,
    seed: int,
    out_checkpoint: Path,
    include_synthetic: bool,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "scripts.train_segmentation",
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--num-workers",
        str(args.num_workers),
        "--case-limit",
        str(args.case_limit),
        "--spatial-size",
        str(args.spatial_size),
        "--num-samples",
        str(args.num_samples),
        "--max-train-batches",
        str(args.max_train_batches),
        "--max-val-batches",
        str(args.max_val_batches),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--out",
        str(out_checkpoint),
    ]
    if include_synthetic:
        cmd.extend(
            [
                "--train-extra-dir",
                args.synthetic_dir,
                "--train-extra-case-limit",
                str(args.dose),
            ]
        )
    if args.quiet_warnings:
        cmd.append("--quiet-warnings")
    if args.no_progress:
        cmd.append("--no-progress")
    return cmd


def build_eval_cmd(
    *,
    args: argparse.Namespace,
    seed: int,
    checkpoint: Path,
    out_metrics: Path,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "scripts.evaluate",
        "--data-dir",
        args.data_dir,
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--out",
        str(out_metrics),
        "--num-workers",
        str(args.num_workers),
        "--case-limit",
        str(args.case_limit),
        "--max-val-batches",
        str(args.max_val_batches),
        "--spatial-size",
        str(args.spatial_size),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--val-ratio",
        str(args.val_ratio),
    ]
    if args.quiet_warnings:
        cmd.append("--quiet-warnings")
    return cmd


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    checkpoint_dir = Path(args.checkpoint_dir)
    metrics_dir = Path(args.metrics_dir)
    tables_dir = Path(args.tables_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    maybe_generate_synthetic_pool(args)

    for seed in seeds:
        baseline_ckpt = checkpoint_dir / f"best_model_cpu_{args.tag}_seed{seed}_baseline.pt"
        synth_ckpt = checkpoint_dir / f"best_model_cpu_{args.tag}_seed{seed}_plus_synth_dose{args.dose}.pt"

        baseline_metrics = metrics_dir / f"seed_{seed}_baseline_{args.tag}.json"
        synth_metrics = metrics_dir / f"seed_{seed}_plus_synth_dose{args.dose}_{args.tag}.json"

        run_step(
            name=f"Seed {seed}: baseline train",
            cmd=build_train_cmd(
                args=args,
                seed=seed,
                out_checkpoint=baseline_ckpt,
                include_synthetic=False,
            ),
            expected_outputs=[baseline_ckpt],
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        run_step(
            name=f"Seed {seed}: baseline eval",
            cmd=build_eval_cmd(
                args=args,
                seed=seed,
                checkpoint=baseline_ckpt,
                out_metrics=baseline_metrics,
            ),
            expected_outputs=[baseline_metrics],
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        run_step(
            name=f"Seed {seed}: synth dose {args.dose} train",
            cmd=build_train_cmd(
                args=args,
                seed=seed,
                out_checkpoint=synth_ckpt,
                include_synthetic=True,
            ),
            expected_outputs=[synth_ckpt],
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        run_step(
            name=f"Seed {seed}: synth dose {args.dose} eval",
            cmd=build_eval_cmd(
                args=args,
                seed=seed,
                checkpoint=synth_ckpt,
                out_metrics=synth_metrics,
            ),
            expected_outputs=[synth_metrics],
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

    baseline_pattern = str(metrics_dir / f"seed_{{seed}}_baseline_{args.tag}.json")
    synth_pattern = str(metrics_dir / f"seed_{{seed}}_plus_synth_dose{args.dose}_{args.tag}.json")
    summary_out = tables_dir / f"seed_ablation_{args.tag}_summary.csv"

    aggregate_cmd = [
        args.python,
        "-m",
        "scripts.aggregate_seed_metrics",
        "--seeds",
        args.seeds,
        "--baseline-pattern",
        baseline_pattern,
        "--variant-pattern",
        synth_pattern,
        "--out",
        str(summary_out),
    ]
    run_step(
        name="Aggregate multiseed summary",
        cmd=aggregate_cmd,
        expected_outputs=[summary_out],
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )

    print("\nCampaign complete.")
    print(f"Summary CSV: {summary_out}")


if __name__ == "__main__":
    main()