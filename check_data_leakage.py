
from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "dataset" / "selected_images.csv"

df = pd.read_csv(MANIFEST_PATH)
df["image_path"] = df["image_path"].astype(str)
df = df[df["image_path"].map(lambda p: Path(p).is_file())].copy()
df = df.drop_duplicates(subset=["id"])

def get_image_hash(path):
    import hashlib
    from PIL import Image

    with Image.open(path) as image:
        return hashlib.sha256(
            image.convert("RGB").tobytes()
        ).hexdigest()

df["image_hash"] = df["image_path"].apply(get_image_hash)

class_names = sorted(df["articleType"].unique().tolist())
class_to_index = {name: i for i, name in enumerate(class_names)}
df["label"] = df["articleType"].map(class_to_index)

train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"],
)

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df["label"],
)

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()

train_df["split"] = "Training"
validation_df["split"] = "Validation"
test_df["split"] = "Testing"

all_splits = pd.concat(
    [train_df, validation_df, test_df]
)

duplicate_groups = all_splits[
    all_splits.duplicated("image_hash", keep=False)
].sort_values("image_hash")

cross_split = duplicate_groups.groupby("image_hash").filter(
    lambda group: group["split"].nunique() > 1
)

print("Duplicate groups found:", duplicate_groups["image_hash"].nunique())
print("Duplicate groups crossing different splits:",
      cross_split["image_hash"].nunique())

if cross_split.empty:
    print("\nNo duplicate images cross between data splits.")
else:
    print("\nPotential data leakage detected:")
    print(
        cross_split[
            ["id", "articleType", "split", "image_path"]
        ].to_string(index=False)
    )

cross_split.to_csv(
    PROJECT_DIR / "results" / "data_leakage_report.csv",
    index=False
)
print("\nReport saved to results/data_leakage_report.csv")
