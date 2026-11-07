
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "dataset" / "myntradataset" / "styles.csv"
IMAGE_DIR = PROJECT_DIR / "dataset" / "myntradataset" / "images"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    "Briefs",
    "Belts",
    "Backpacks",
    "Socks",
    "Formal Shoes",
]

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

required_columns = {"id", "articleType"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing required columns. Found: {df.columns.tolist()}"
    )

df = df[df["articleType"].isin(CATEGORIES)].copy()

df["image_path"] = df["id"].apply(
    lambda product_id: IMAGE_DIR / f"{product_id}.jpg"
)
df = df[df["image_path"].apply(lambda path: path.is_file())].copy()

df = df.drop_duplicates(subset=["id"]).copy()

def get_image_hash(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(image.tobytes()).hexdigest()

df["image_hash"] = df["image_path"].apply(get_image_hash)

duplicate_rows = df[
    df.duplicated(subset=["image_hash"], keep="first")
].copy()

duplicate_report_path = RESULTS_DIR / "removed_duplicates.csv"
duplicate_rows.to_csv(duplicate_report_path, index=False)

df = df.drop_duplicates(
    subset=["image_hash"],
    keep="first"
).copy()

df = df.drop(columns=["image_hash"])

print("\nFinal verified image counts by category:")
print(df["articleType"].value_counts().reindex(CATEGORIES, fill_value=0))

print("\nTotal images before duplicate removal:", 
      len(df) + len(duplicate_rows))
print("Duplicate images removed:", len(duplicate_rows))
print("Final unique images:", len(df))

output_path = PROJECT_DIR / "dataset" / "selected_images.csv"

df[["id", "articleType", "image_path"]].to_csv(
    output_path,
    index=False
)

print(f"\nSaved clean manifest to: {output_path}")
print(f"Duplicate removal report: {duplicate_report_path}")
