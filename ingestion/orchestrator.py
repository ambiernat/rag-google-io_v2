from pathlib import Path
import yaml
from fetch import main as fetch_main
from chunk import main as chunk_main
from canonicalize import main as canonicalize_main

# --------------------
# Paths & Config
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "ingestion.yaml"

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# --------------------
# Pipeline Orchestration
# --------------------
def main():
    config = load_config(CONFIG_PATH)
    
    overwrite = config.get("overwrite", False)

    print("\n=== Step 1: Fetch transcripts ===")
    fetch_main(
        video_ids=config["videos"],
        languages=config.get("languages", ["en"]),
        overwrite=overwrite
    )

    print("\n=== Step 2: Chunk transcripts ===")
    chunk_main(overwrite=overwrite)

    print("\n=== Step 3: Canonicalize chunks ===")
    canonicalize_main(overwrite=overwrite)

    print("\n=== Ingestion pipeline complete ===")

# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    main()
