# Real-Time Cybersecurity Threat Detection

An end-to-end machine learning pipeline for detecting potential network security threats using the CICIDS2017 dataset. This project combines data engineering, supervised/unsupervised machine learning, MLflow experiment tracking, Apache Kafka streaming, and FastAPI to build a real-time threat detection system.

## Architecture

```text
Network Traffic Events
        |
        v
Kafka Producer
        |
        v
Kafka: security-events
        |
        v
Python Consumer
        |
        v
Random Forest Model
(loaded through MLflow)
        |
        v
Threat Prediction
        |
        v
Kafka: threat-predictions
```

FastAPI provides an additional HTTP interface for direct model inference.

## Tech Stack

- **Python:** Data processing and machine learning
- **Pandas & NumPy:** Data ingestion, preprocessing, and analysis
- **Scikit-learn:** Random Forest classification
- **MLflow:** Experiment tracking and model management
- **Apache Kafka:** Real-time event streaming
- **FastAPI:** REST API for model inference
- **Docker & Docker Compose:** Containerization and service orchestration

## Dataset

This project uses the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html), which contains labeled network traffic records collected over five days, including benign traffic and multiple categories of network attacks.

### Download and Setup

1. Download the original CICIDS2017 dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Extract the downloaded files.
3. Place the original daily CSV files in the following directory:

```text
data/
└── raw/
    └── cicids2017/
        ├── Monday-WorkingHours.pcap_ISCX.csv
        ├── Tuesday-WorkingHours.pcap_ISCX.csv
        ├── ...
        └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

4. Run the data ingestion and preprocessing scripts to validate, clean, and split the raw data into training, validation, and test datasets.

The processed directories are generated automatically by the ingestion and preprocessing pipeline:

```text
data/
├── raw/
│   └── cicids2017/
└── processed/
    ├── train/
    ├── validation_rev/
    └── test_rev/
```

**Note:** The raw and processed datasets are not included in this repository. They are generated locally after downloading the original dataset and running the pipeline.

## Data Engineering: Ingestion, Preprocessing & Validation

Before model development, I implemented a structured data preparation workflow to inspect, audit, ingest, split, and validate the raw CICIDS2017 dataset.

### 1. Initial Dataset Inspection (`inspect_dataset.py`)

Inspected the original CSV files to understand their structure and identify potential schema inconsistencies.

- Discovered all raw CSV files.
- Sampled the first five rows of each file.
- Examined column names and column counts.
- Compared column names and their order across files.
- Identified potential target label columns.

### 2. Raw Dataset Audit (`audit_dataset.py`)

Developed a chunk-based auditing script to understand the size and class distribution of the original dataset.

- Processed files in chunks of 100,000 rows to minimize memory usage.
- Counted total records, benign records, and attack records.
- Calculated attack rates and individual attack label frequencies.
- Exported a consolidated audit report to `data/raw/audit_summary.csv`.

### 3. Initial Data Ingestion (`ingest.py`)

Implemented an initial ingestion pipeline using pandas chunk processing.

- Processed raw CSV files in chunks of 100,000 rows.
- Standardized column names and target label values by removing surrounding whitespace.
- Converted the original multiclass `Label` column into a binary classification target:
  - `BENIGN` → 0
  - All attack categories → 1
- Created initial training, validation, and test datasets.

### 4. Revised Dataset Splitting (`ingest_data_revised_split.py`)

During initial model validation, the chronological split resulted in a very low attack representation in the validation dataset, making it less suitable for model selection and decision-threshold tuning.

To address this, the dataset splitting strategy was revised to create a more representative validation dataset while retaining a separate test dataset for final evaluation.

The revised split was:

- **Training data:** Monday, Tuesday, and Wednesday traffic.
- **Validation data:** Thursday Web Attacks, Thursday Infiltration, and Friday Morning traffic.
- **Test data:** Friday Afternoon DDoS and PortScan traffic only.

The revised split was used for subsequent model development, model selection, and decision-threshold tuning. The test dataset was held out for final model evaluation.

### 5.  Processed Data Validation (`validate_processed_data.py`)

Developed a validation script to assess the quality and consistency of the ingested datasets before model training.

The script:

- Calculates row counts for individual files and each dataset split.
- Examines benign and attack distributions.
- Calculates attack rates for each split.
- Checks and aggregates missing values by column.
- Compares numeric and non-numeric data type consistency across processed files.

### Processed Data Organization

The original dataset is kept separate from the processed data.

```text
data/
├── raw/
│   ├── cicids2017/
│   └── audit_summary.csv
│
└── processed/
    ├── train/
    ├── validation_rev/
    └── test_rev/
```

The processed datasets are used for feature preprocessing, model training, validation, and testing.

## Machine Learning


### Custom Feature Preprocessing (`preprocessing.py`)

Implemented a custom scikit-learn-compatible transformer, `CICIDSPreprocessor`, using `BaseEstimator` and `TransformerMixin`.

The transformer is responsible for initial feature preprocessing and input validation within the machine learning pipeline.

**Feature selection and cleanup:**
- Removes duplicate, redundant, and non-informative features identified during dataset exploration.
- Maintains a consistent feature schema for model training and inference.

**Feature schema management:**
- Learns the expected feature names and their order during `fit()`.
- Checks that all required features are present during `transform()`.
- Ignores additional features that are not part of the trained model's feature set.
- Preserves the original training feature order.

**Input validation:**
- Rejects inputs containing missing values (NaN).
- Rejects inputs containing infinite values.
- Raises an error when required features are missing.

The custom transformer is designed to be integrated into the scikit-learn machine learning pipeline, ensuring consistent preprocessing and input validation during both model training and inference.


### Model Training (`train.py`)

Implemented a Random Forest training and evaluation workflow.

- Loads the processed training and validation datasets.
- Trains the Random Forest classifier.
- Configures model hyperparameters.
- Evaluates model performance using classification metrics.
- Tunes the decision threshold using validation predictions.
- Logs the trained model, parameters, metrics, and selected decision threshold to MLflow.

The final trained model is retrieved from MLflow for downstream inference through FastAPI and the Kafka consumer.

### Model Evaluation

The model is evaluated using classification metrics to assess its ability to distinguish benign traffic from malicious network activity.

The decision threshold is selected using validation data rather than relying on the default classification threshold of 0.5.

## MLflow: Experiment Tracking and Model Management

MLflow is used to manage the machine learning lifecycle.

- Tracks experiments and model parameters.
- Logs trained Random Forest models.
- Records the selected decision threshold.
- Loads the trained model for inference.

The inference service retrieves the trained model and its associated threshold from MLflow.

## Real-Time Streaming with Apache Kafka

Apache Kafka provides the streaming infrastructure for processing network events and distributing model predictions.

Two Kafka topics are used:

| Topic | Purpose |
|---|---|
| `security-events` | Receives incoming network traffic events |
| `threat-predictions` | Publishes model predictions |

### Producer

The Python producer publishes network traffic events to the `security-events` topic.

### Consumer

The Python consumer:

1. Listens for incoming events from `security-events`.
2. Passes event features to the inference component.
3. Uses the trained Random Forest model to generate predictions.
4. Generates an attack probability and predicted label.
5. Publishes the resulting prediction to `threat-predictions` topic.

This separates event ingestion from model inference and prediction distribution.

## FastAPI

FastAPI provides an HTTP interface for model inference.

It allows external applications and services to submit network traffic features and receive structured prediction responses without directly interacting with the underlying model.

Interactive API documentation is available through Swagger UI.

Example prediction response:

```json
{
  "prediction": 0,
  "label": "Benign",
  "attack_probability": 0.045
}
```

## Project Structure

```text
cyber-threat-detection/
│
├── data/
│   ├── raw/
│   │   ├── cicids2017/
│   │   └── audit_summary.csv
│   │
│   └── processed/
│       ├── train/
│       ├── validation_rev/
│       └── test_rev/
│
├── notebooks/
│
├── src/
│   ├── inspect_dataset.py
│   ├── audit_dataset.py
│   ├── ingest.py
│   ├── ingest_data_revised_split.py
│   ├── preprocessing.py
│   ├── validate_processed_data.py
│   ├── train.py
│   ├── api.py
│   ├── producer.py
│   ├── consumer.py
│   └── inference.py
│
├── docker/
├── models/
├── mlruns/
├── mlflow.db
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Prerequisites

- Docker Desktop
- Python 3.11+
- Git

## Setup

Clone the repository:

```bash
git clone <your-repository-url>
cd cyber-threat-detection
```

Create and activate a virtual environment:

```bash
python -m venv tf_env
source tf_env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the CICIDS2017 dataset and place the original CSV files under:

```text
data/raw/cicids2017/
```

### Data Preparation

Run the inspection and audit scripts:

```bash
python src/inspect_dataset.py
python src/audit_dataset.py
```

Run the revised ingestion, preprocessing, and validation workflow:

```bash
python src/ingest_data_revised_split.py
python src/validate_processed_data.py
```

### Feature Preprocessing and Model Training

After preparing the datasets, train the Random Forest model:

```bash
python src/train.py
```

The trained model and associated parameters are logged to MLflow.

## Run the Application

Build and start the Docker Compose services:

```bash
docker compose up --build -d
```

Check running containers:

```bash
docker compose ps
```

View consumer logs:

```bash
docker compose logs -f consumer
```

View FastAPI logs:

```bash
docker compose logs -f fastapi
```

### FastAPI Endpoints

Once the services are running:

- API: http://localhost:8000
- Interactive API documentation: http://localhost:8000/docs

## Future Improvements

- Improve attack detection recall while maintaining acceptable precision.
- Strengthen input validation and error handling.
- Expand model evaluation and monitoring.
- Explore advanced anomaly detection and agentic AI workflows.

## License

This project is intended for educational and portfolio purposes.
