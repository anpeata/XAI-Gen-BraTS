from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

MODALITIES = ("t1", "t1ce", "t2", "flair")

SYNAPSE_MODALITY_MAP = {
    "t1": "t1n",
    "t1ce": "t1c",
    "t2": "t2w",
    "flair": "t2f",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare BraTS dataset into data/processed/BraTS2023"
    )
    p.add_argument("--source", choices=["none", "kaggle", "huggingface"], default="none")
    p.add_argument("--kaggle-dataset", type=str, default="", help="owner/dataset-slug")
    p.add_argument("--kaggle-competition", type=str, default="", help="competition slug")
    p.add_argument("--hf-repo-id", type=str, default="", help="dataset repo id")
    p.add_argument("--raw-dir", type=str, default="data/raw/BraTS2023")
    p.add_argument("--processed-dir", type=str, default="data/processed/BraTS2023")
    p.add_argument("--extract-zips", action="store_true")
    p.add_argument("--link-mode", choices=["copy", "hardlink"], default="copy")
    p.add_argument("--limit", type=int, default=0, help="optional max number of cases to prepare")
    p.add_argument(
        "--include-case-prefix",
        nargs="*",
        default=[],
        help="optional case prefixes to include, e.g. BraTS-GLI BraTS-MEN",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def download_from_kaggle(raw_dir: Path, dataset_slug: str, competition_slug: str) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("kaggle package is not installed. Install requirements first.") from exc

    api = KaggleApi()
    api.authenticate()

    if dataset_slug:
        print(f"Downloading Kaggle dataset: {dataset_slug}")
        api.dataset_download_files(dataset_slug, path=str(raw_dir), unzip=False, quiet=False)
    elif competition_slug:
        print(f"Downloading Kaggle competition files: {competition_slug}")
        api.competition_download_files(competition_slug, path=str(raw_dir), quiet=False)
    else:
        raise ValueError("For Kaggle source, provide --kaggle-dataset or --kaggle-competition")


def download_from_huggingface(raw_dir: Path, repo_id: str) -> None:
    if not repo_id:
        raise ValueError("For Hugging Face source, provide --hf-repo-id")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub package is not installed. Install requirements first.") from exc

    print(f"Downloading Hugging Face dataset: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.nii.gz", "*.nii", "*.zip", "*.csv", "*.json", "*.txt"],
    )


def extract_zips(root: Path) -> None:
    zip_files = sorted(root.rglob("*.zip"))
    if not zip_files:
        print("No zip archives found.")
        return

    for zip_path in zip_files:
        out_dir = zip_path.with_suffix("")
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"Skipping extraction (already populated): {out_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path} -> {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, out_dir)


def extract_tars(root: Path) -> None:
    tar_files = sorted(root.rglob("*.tar.gz"))
    if not tar_files:
        print("No tar.gz archives found.")
        return

    for tar_path in tar_files:
        out_dir = tar_path.parent / tar_path.name.replace(".tar.gz", "")
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"Skipping extraction (already populated): {out_dir}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extract_tar(tf, out_dir)


def _is_within_directory(root: Path, target: Path) -> bool:
    root_resolved = root.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    try:
        target_resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def _safe_extract_zip(zf: zipfile.ZipFile, out_dir: Path) -> None:
    out_dir_resolved = out_dir.resolve(strict=False)
    for info in zf.infolist():
        dest = out_dir / info.filename
        if not _is_within_directory(out_dir_resolved, dest):
            raise RuntimeError(f"Unsafe path in zip archive: {info.filename}")
    zf.extractall(out_dir)


def _safe_extract_tar(tf: tarfile.TarFile, out_dir: Path) -> None:
    out_dir_resolved = out_dir.resolve(strict=False)
    for member in tf.getmembers():
        dest = out_dir / member.name
        if not _is_within_directory(out_dir_resolved, dest):
            raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
    tf.extractall(out_dir)


def filter_cases_by_prefix(
    cases: list[tuple[str, dict[str, Path]]],
    include_prefixes: list[str],
) -> list[tuple[str, dict[str, Path]]]:
    if not include_prefixes:
        return cases

    prefixes = tuple(include_prefixes)
    return [(case_id, files) for case_id, files in cases if case_id.startswith(prefixes)]


def discover_cases(search_root: Path) -> list[tuple[str, dict[str, Path]]]:
    cases: list[tuple[str, dict[str, Path]]] = []
    for seg_path in sorted(search_root.rglob("*_seg.nii.gz")):
        case_id = seg_path.name.replace("_seg.nii.gz", "")
        files: dict[str, Path] = {"seg": seg_path}
        ok = True
        for mod in MODALITIES:
            p = seg_path.with_name(f"{case_id}_{mod}.nii.gz")
            if not p.exists():
                ok = False
                break
            files[mod] = p
        if ok:
            cases.append((case_id, files))
    return cases


def discover_cases_synapse(search_root: Path) -> list[tuple[str, dict[str, Path]]]:
    """Discover BraTS cases from Synapse naming:

    - images: <case>-t1n/t1c/t2w/t2f.nii.gz under data/<case>/
    - labels: <case>-seg.nii.gz under labels/
    """
    cases: list[tuple[str, dict[str, Path]]] = []
    for seg_path in sorted(search_root.rglob("*-seg.nii.gz")):
        case_id = seg_path.name.replace("-seg.nii.gz", "")
        files: dict[str, Path] = {"seg": seg_path}

        ok = True
        for mod in MODALITIES:
            syn_mod = SYNAPSE_MODALITY_MAP[mod]
            pattern = f"{case_id}-{syn_mod}.nii.gz"
            candidates = list(search_root.rglob(pattern))
            if not candidates:
                ok = False
                break
            files[mod] = candidates[0]

        if ok:
            cases.append((case_id, files))
    return cases


def _transfer(src: Path, dst: Path, link_mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    if dst.exists():
        return

    if link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)


def prepare_processed_split(
    cases: list[tuple[str, dict[str, Path]]],
    processed_dir: Path,
    link_mode: str,
    limit: int,
    dry_run: bool,
) -> int:
    count = 0
    for case_id, files in cases:
        if limit > 0 and count >= limit:
            break

        case_dir = processed_dir / case_id
        if not dry_run:
            case_dir.mkdir(parents=True, exist_ok=True)

        names = ["seg", *MODALITIES]
        for name in names:
            suffix = "seg" if name == "seg" else name
            dst = case_dir / f"{case_id}_{suffix}.nii.gz"
            _transfer(files[name], dst, link_mode=link_mode, dry_run=dry_run)

        count += 1
    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "kaggle":
        download_from_kaggle(raw_dir, args.kaggle_dataset, args.kaggle_competition)
    elif args.source == "huggingface":
        download_from_huggingface(raw_dir, args.hf_repo_id)

    if args.extract_zips:
        extract_zips(raw_dir)
        extract_tars(raw_dir)

    cases = discover_cases(raw_dir)
    if not cases:
        cases = discover_cases_synapse(raw_dir)

    cases = filter_cases_by_prefix(cases, args.include_case_prefix)
    if not cases:
        raise RuntimeError(
            "No valid BraTS cases found. Ensure files include *_t1, *_t1ce, *_t2, *_flair, *_seg as .nii.gz"
        )

    prepared = prepare_processed_split(
        cases,
        processed_dir=processed_dir,
        link_mode=args.link_mode,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print(f"Discovered cases: {len(cases)}")
    print(f"Prepared cases: {prepared}")
    print(f"Processed directory: {processed_dir}")


if __name__ == "__main__":
    main()
