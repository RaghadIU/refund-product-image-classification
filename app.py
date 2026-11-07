
from pathlib import Path
import io

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "refund_classifier.keras"

IMAGE_SIZE = (160, 160)

CLASS_NAMES = [
    "Backpacks",
    "Belts",
    "Briefs",
    "Formal Shoes",
    "Socks",
]

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_bytes):
    """Return the predicted category and probabilities for one image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")

    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_category": CLASS_NAMES[predicted_index],
        "probabilities": {
            name: round(float(probability), 6)
            for name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


@app.get("/")
def home():
    return jsonify({
        "service": "Refund Product Image Classification API",
        "status": "running",
        "classes": CLASS_NAMES,
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "user_interface": "/ui",
        },
    })


@app.get("/ui")
def user_interface():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict_single():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    try:
        result = predict_image(request.files["image"].read())
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/predict/batch")
def predict_batch():
    files = request.files.getlist("images")

    if not files:
        return jsonify({
            "error": "Upload one or more images using the 'images' field."
        }), 400

    results = []

    for uploaded_file in files:
        try:
            prediction = predict_image(uploaded_file.read())
            results.append({
                "filename": uploaded_file.filename,
                **prediction,
            })
        except ValueError as error:
            results.append({
                "filename": uploaded_file.filename,
                "error": str(error),
            })

    return jsonify({
        "total_images": len(files),
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
