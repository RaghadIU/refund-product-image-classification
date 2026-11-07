
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# 1. Paths and settings

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "dataset" / "selected_images.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)



# 2. Load and validate dataset
df = pd.read_csv(MANIFEST_PATH)

df["image_path"] = df["image_path"].astype(str)
df = df[df["image_path"].map(lambda p: Path(p).is_file())].copy()
df = df.drop_duplicates(subset=["id"])

class_names = sorted(df["articleType"].unique().tolist())
class_to_index = {
    name: index for index, name in enumerate(class_names)
}

df["label"] = df["articleType"].map(class_to_index)

print("Classes:", class_names)
print("\nImages per class:")
print(df["articleType"].value_counts())
print("\nTotal usable images:", len(df))

if len(class_names) < 2:
    raise ValueError("At least two classes are required.")

if df["articleType"].value_counts().min() < 2:
    raise ValueError("Each class needs at least two images.")


# 3. Split into training, validation, and test sets
train_df, temporary_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"],
)

validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temporary_df["label"],
)

print("\nTraining images:", len(train_df))
print("Validation images:", len(validation_df))
print("Test images:", len(test_df))


# 4. Build image-loading pipelines
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    return image, label


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
validation_ds = make_dataset(validation_df)
test_ds = make_dataset(test_df)



# 5. Create transfer-learning model
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.20)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



# 6. Train and save model
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
)

model_path = MODEL_DIR / "refund_classifier.keras"
model.save(model_path)

print(f"\nModel saved to: {model_path}")



# 7. Evaluate on the held-out test set
y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

probabilities = model.predict(test_ds)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
)

results = {
    "classes": class_names,
    "total_images": int(len(df)),
    "training_images": int(len(train_df)),
    "validation_images": int(len(validation_df)),
    "test_images": int(len(test_df)),
    "test_accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
}

results_path = RESULTS_DIR / "evaluation.json"
with open(results_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

history_path = RESULTS_DIR / "training_history.json"
with open(history_path, "w", encoding="utf-8") as file:
    json.dump(history.history, file, indent=2)

print(f"\nEvaluation results saved to: {results_path}")
print(f"Training history saved to: {history_path}")
