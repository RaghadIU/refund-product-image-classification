import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset" / "selected_images.csv"
INCOMING_DIR = BASE_DIR / "data" / "incoming"
RESULTS_DIR = BASE_DIR / "results"
USED_CSV = RESULTS_DIR / "used_images.csv"

# Number of new images per daily run
DAILY_IMAGE_LIMIT = 500

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_image_path(image_path):
    """Find an image using its saved path or known dataset folders."""
    path = Path(str(image_path))

    if path.is_file():
        return path

    # If the saved path is no longer valid, search by filename.
    filename = path.name
    possible_locations = [
        BASE_DIR / "dataset" / "myntradataset" / "images" / filename,
        BASE_DIR / "dataset" / "images" / filename,
    ]

    for candidate in possible_locations:
        if candidate.is_file():
            return candidate

    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"id", "articleType", "image_path"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in selected_images.csv: {sorted(missing_columns)}"
        )

    # Load the registry of images used in previous automated runs.
    if USED_CSV.exists():
        used_df = pd.read_csv(USED_CSV)
    else:
        used_df = pd.DataFrame(
            columns=["id", "articleType", "source_path", "copied_filename", "used_at"]
        )

    used_ids = set(used_df["id"].astype(str)) if not used_df.empty else set()

    # On the first run, exclude numeric image names already in incoming.
    # This avoids reusing the 10 images you copied there for earlier tests.
    for existing_file in INCOMING_DIR.iterdir():
        if existing_file.is_file() and existing_file.suffix.lower() in SUPPORTED_EXTENSIONS:
            if existing_file.stem.isdigit():
                used_ids.add(existing_file.stem)

    # Select only eligible images that have not been used before.
    df["id_key"] = df["id"].astype(str)
    candidates = df[~df["id_key"].isin(used_ids)].copy()

    # Resolve paths before selecting images so missing files are skipped.
    candidates["resolved_path"] = candidates["image_path"].apply(resolve_image_path)
    candidates = candidates[candidates["resolved_path"].notna()].copy()

    if candidates.empty:
        print("No unused eligible images with valid file paths remain.")
        return

    count = min(DAILY_IMAGE_LIMIT, len(candidates))

    # Shuffle the candidates so daily batches are not always in CSV order.
    selected = candidates.sample(n=count, random_state=None)

    # Remove old image files from incoming so today's batch is processed alone.
    for existing_file in INCOMING_DIR.iterdir():
        if existing_file.is_file() and existing_file.suffix.lower() in SUPPORTED_EXTENSIONS:
            existing_file.unlink()

    used_at = datetime.now().astimezone().isoformat(timespec="seconds")
    new_records = []

    for _, row in selected.iterrows():
        source = Path(row["resolved_path"])
        image_id = str(row["id"])
        destination_name = f"{image_id}{source.suffix.lower()}"
        destination = INCOMING_DIR / destination_name

        # The selected IDs are unique, so destination names should be unique.
        shutil.copy2(source, destination)

        new_records.append({
            "id": image_id,
            "articleType": row["articleType"],
            "source_path": str(source),
            "copied_filename": destination_name,
            "used_at": used_at,
        })

    new_used_df = pd.DataFrame(new_records)
    updated_used_df = pd.concat([used_df, new_used_df], ignore_index=True)
    updated_used_df.to_csv(USED_CSV, index=False)

    print(f"Eligible images available for this run: {len(candidates)}")
    print(f"New images copied to incoming: {len(new_records)}")
    print(f"Used-image registry saved to: {USED_CSV}")


if __name__ == "__main__":
    main()