from pathlib import Path

def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(gt_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No ground truth files with prefix '{prefix}' found in {gt_dir}")
    return files[0]
