# Refund Product Image Classification System

## 1. Project Overview

This project develops an automated product image classification system for a refund department. It uses deep learning to classify product images into five predefined categories.

The project focuses on **batch processing**, allowing the system to automatically process multiple product images and monitor the prediction results.

The system is designed to support product image classification. It does not determine refund eligibility or detect product damage.

## 2. Project Objectives

- Develop a deep learning model for product image classification.
- Classify images into five predefined product categories.
- Integrate the trained model into a Flask REST API.
- Support single-image and batch predictions.
- Automate nightly batch processing.
- Monitor prediction results, confidence scores, and processing errors.

## 3. Product Categories

The model supports five categories:

1. Briefs
2. Belts
3. Backpacks
4. Socks
5. Formal Shoes

## 4. Dataset

The project uses the Fashion Product Images dataset from Kaggle.

- Selected images: 3,703
- Training set: 70%
- Validation set: 15%
- Testing set: 15%

### Data Preparation

The dataset preparation process included:

- Selecting images from the five target categories.
- Removing six excess exact duplicate images.
- Checking image duplicates using SHA-256 hashing.
- Splitting the clean dataset into training, validation, and testing sets.

## 5. Model Development

The classification model was developed using **MobileNetV2** with transfer learning.

### Model Configuration

- Architecture: MobileNetV2
- Input image size: 160 × 160
- Batch size: 32
- Training epochs: 5
- Pre-trained base model: Frozen
- Output layer: Softmax
- Number of classes: 5

### Model Evaluation

The model achieved approximately **99.82% test accuracy** on a separate test set of 556 images.

Evaluation also included precision, recall, and F1-score.

## 6. System Architecture

The system consists of the following components:

1. Dataset preparation and cleaning.
2. Deep learning model training.
3. Flask REST API integration.
4. Single-image and batch prediction.
5. Automated nightly processing.
6. Monitoring and logging.

## 7. REST API

The trained model is integrated into a Flask application.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API availability |
| POST | `/predict` | Predicts a single image |
| POST | `/predict/batch` | Predicts multiple images |
| GET | `/ui` | Opens the graphical interface |

The prediction response includes the predicted category and probability scores.

### Graphical User Interface (GUI)

The project includes a web-based graphical interface that allows users to upload product images and view classification results.

The interface was successfully tested. It displays:
- Predicted product category.
- Prediction confidence score.
- Probability scores for all five categories.

Local interface: http://127.0.0.1:5000/ui

## 8. Batch Processing

The batch processing pipeline automatically:

1. Selects up to 500 unused images.
2. Copies the selected images into the incoming folder.
3. Sends the images to the Flask API.
4. Generates predictions for each image.
5. Saves the results in timestamped CSV files.

## 9. Monitoring and Logging

The monitoring process tracks:

- Batch success rate.
- Number of processing errors.
- Average prediction confidence.
- Number of predictions with confidence below 60%.

Monitoring results are stored in a JSONL log, while nightly execution details are recorded in a separate log file.

## 10. Automation

The nightly batch process is automated using **Windows Task Scheduler**.

The scheduled task runs the `run_nightly.py` script daily at 10:00 PM.

The script coordinates image preparation, API availability, batch prediction, and monitoring.

## 11. Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Flask
- Pandas
- NumPy
- Pillow
- REST API
- Windows Task Scheduler
- CSV and JSONL

## 12. Project Structure

The main project components include:

```text
refund-classification/
│
├── app.py
├── batch_predict.py
├── prepare_daily_images.py
├── monitoring.py
├── run_nightly.py
│
├── dataset/
├── data/
│   └── incoming/
│
├── models/
│   └── refund_classifier.keras
│
├── results/
│   ├── evaluation.json
│   ├── used_images.csv
│   ├── monitoring_log.jsonl
│   └── nightly_run.log
│
└── templates/
    └── index.html
```

*Note: This is a simplified overview of the main project files and folders.*

## 13. Running the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd refund-classification
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Step 4: Start the Flask Application

```bash
python app.py
```

### Step 5: Open the Graphical Interface

Open the following address in your browser:

```text
http://127.0.0.1:5000/ui
```

### Step 6: Run the Nightly Batch Process

```bash
python run_nightly.py
```

## 14. Limitations

- The model supports only five predefined product categories.
- Images outside the supported categories may still be assigned to one of the five classes.
- The system classifies product categories but does not detect damage or defects.
- Model performance depends on the quality and variety of the training dataset.

## 15. Future Improvements

- Expand the dataset with more product categories.
- Improve predictions for unfamiliar product images.
- Deploy the system to a cloud environment.
- Develop advanced monitoring dashboards and reports.
