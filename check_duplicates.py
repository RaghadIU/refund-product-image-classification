
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "dataset" / "selected_images.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(MANIFEST_PATH)

def get_image_hash(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        content = image.tobytes()
        return hashlib.sha256(content).hexdigest()

print("Checking images for exact duplicates...")

df["image_hash"] = df["image_path"].apply(get_image_hash)

duplicate_mask = df.duplicated(
    subset=["image_hash"], keep=False
)

duplicates = df[duplicate_mask].sort_values("image_hash")

report_path = RESULTS_DIR / "duplicate_report.csv"
duplicates.to_csv(report_path, index=False)

duplicate_groups = duplicates["image_hash"].nunique()

print("\nDuplicate check completed.")
print("Total images checked:", len(df))
print("Images involved in duplicates:", len(duplicates))
print("Duplicate groups:", duplicate_groups)
print("Report saved to:", report_path)

if duplicates.empty:
    print("\nNo exact duplicate images found.")
else:
    print("\nDuplicate images were found.")
    print(duplicates[["id", "articleType", "image_path"]].to_string(index=False))
