# ingestion/fetch.py
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
import json
from pathlib import Path
from datetime import datetime, timezone
import yaml

# --------------------
# Paths
# --------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_PATH = BASE_DIR / "configs" / "ingestion.yaml"
MANIFEST_PATH = RAW_DIR / "_manifest.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_SCHEMA_VERSION = "raw_v1"


# --------------------
# Config
# --------------------
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    
# --------------------
# Manifest
# --------------------

def log_manifest(entry: dict) -> None:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    manifest.append(entry)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

# --------------------
# Fetch logic
# --------------------

api = YouTubeTranscriptApi()

def fetch_transcript(api, video_id: str, languages=("en",)):
    transcripts = api.list(video_id)
    transcript = transcripts.find_transcript(languages)

    segments = [
        {
            "text": item.text,
            "start": item.start,
            "duration": item.duration,
        }
        for item in transcript.fetch()
    ]

    return segments, transcript.language_code


# --------------------
# Normalize raw schema
# --------------------
def build_raw_payload(
    video_id: str,
    language: str,
    segments: list[dict],
) -> dict:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "video_id": video_id,
        "language": language,
        "segments": segments,
    }

# --------------------
# Main
# --------------------
def main(video_ids: list[str], languages: list[str], overwrite: bool = False) -> None:
    languages = tuple(languages)

    for video_id in video_ids:
        out_path = RAW_DIR / f"{video_id}.json"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {video_id} already exists")
            continue

        try:
            segments, language = fetch_transcript(api, video_id, languages)

            payload = build_raw_payload(
                video_id=video_id,
                language=language,
                segments=segments,
            )

            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

            log_manifest(
                {
                    "video_id": video_id,
                    "language": language,
                    "status": "ok",
                    "schema_version": RAW_SCHEMA_VERSION,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            print(f"[OK] Fetched transcript for {video_id}")

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            log_manifest(
                {
                    "video_id": video_id,
                    "language": languages[0],
                    "status": "no_transcript",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"[NO TRANSCRIPT] {video_id}: {e}")

        except Exception as e:
            log_manifest(
                {
                    "video_id": video_id,
                    "language": languages[0],
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"[ERROR] {video_id}: {e}")


# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    config = load_config(CONFIG_PATH)

    main(
        video_ids=config["videos"],
        languages=config.get("languages", ["en"]),
        overwrite=config.get("overwrite", False),
    )