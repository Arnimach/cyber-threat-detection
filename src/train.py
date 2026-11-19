
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

from src.preprocessing import CICIDSPreprocessor


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data/processed/train"
VAL_PATH = PROJECT_ROOT / "data/processed/validation_rev"
TEST_PATH = PROJECT_ROOT / "data/processed/test_rev"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlruns"

MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

EXPERIMENT_NAME = "cyber-threat-detection-random-forest-model"
RUN_NAME = "random-forest-fine-tuned"

THRESHOLD = 0.31929

MODEL_NAME = "random_forest_model"


RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 30,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_dataset(files):

    if not files:
        raise FileNotFoundError("No CSV files found.")

    dataframes = [
        pd.read_csv(file)
        for file in files
    ]

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    y = combined_df["target"]

    X = combined_df.drop(
        columns=["target", "Label"],
        errors="ignore"
    )

    return X, y


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_training_data(X, y, preprocessor):

    X_features = X.drop(
        columns=preprocessor.columns_to_drop,
        errors="ignore"
    )

    missing = (
        set(preprocessor.feature_names_)
        - set(X_features.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    X_features = X_features.loc[
        :, preprocessor.feature_names_
    ]

    negative_cols = [
        col for col in preprocessor.negative_columns
        if col in X_features.columns
    ]

    valid_mask = ~(
        X_features[negative_cols] < 0
    ).any(axis=1)

    valid_mask &= np.isfinite(
        X_features.to_numpy(dtype=float)
    ).all(axis=1)

    valid_mask &= X_features.notna().all(axis=1)

    return X.loc[valid_mask], y.loc[valid_mask]


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X, y, threshold):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    return {
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "f2": fbeta_score(
            y, predictions, beta=2, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions
        ).tolist(),
    }


# --------------------------------------------------
# MLFLOW CONFIGURATION
# --------------------------------------------------

def configure_mlflow():

    client = MlflowClient()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:

        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=(
                MLFLOW_ARTIFACTS.resolve().as_uri()
            )
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train_files = sorted(TRAIN_PATH.glob("*.csv"))
    validation_files = sorted(VAL_PATH.glob("*.csv"))
    test_files = sorted(TEST_PATH.glob("*.csv"))

    X_train, y_train = load_dataset(train_files)
    X_val, y_val = load_dataset(validation_files)
    X_test, y_test = load_dataset(test_files)

    print("Establishing feature schema...")

    preprocessor = CICIDSPreprocessor()
    preprocessor.fit(X_train)

    print("Cleaning datasets...")

    X_train_clean, y_train_clean = clean_training_data(
        X_train, y_train, preprocessor
    )

    X_val_clean, y_val_clean = clean_training_data(
        X_val, y_val, preprocessor
    )

    X_test_clean, y_test_clean = clean_training_data(
        X_test, y_test, preprocessor
    )

    print("Building Random Forest pipeline...")

    pipeline = Pipeline([
        ("preprocessor", CICIDSPreprocessor()),
        (
            "classifier",
            RandomForestClassifier(**RF_PARAMS)
        )
    ])

    print("Training model...")

    pipeline.fit(X_train_clean, y_train_clean)

    print("Evaluating model...")

    val_metrics = evaluate_model(
        pipeline,
        X_val_clean,
        y_val_clean,
        THRESHOLD
    )

    test_metrics = evaluate_model(
        pipeline,
        X_test_clean,
        y_test_clean,
        THRESHOLD
    )

    # Configure MLflow
    configure_mlflow()

    print("Logging model and experiment to MLflow...")

    with mlflow.start_run(run_name=RUN_NAME):

        mlflow.log_params(RF_PARAMS)

        mlflow.log_param(
            "decision_threshold",
            THRESHOLD
        )

        mlflow.log_param(
            "training_rows",
            len(X_train_clean)
        )

        mlflow.log_param(
            "validation_rows",
            len(X_val_clean)
        )

        mlflow.log_param(
            "test_rows",
            len(X_test_clean)
        )

        mlflow.log_param(
            "number_of_features",
            len(preprocessor.feature_names_)
        )

        for name, metrics in [
            ("validation", val_metrics),
            ("test", test_metrics)
        ]:

            for metric_name, value in metrics.items():

                if isinstance(value, (int, float)):
                    mlflow.log_metric(
                        f"{name}_{metric_name}",
                        value
                    )

            mlflow.log_dict(
                {
                    "confusion_matrix":
                        metrics["confusion_matrix"]
                },
                f"{name}_confusion_matrix.json"
            )

        mlflow.log_dict(
            {
                "feature_names":
                    preprocessor.feature_names_
            },
            "feature_names.json"
        )

        mlflow.log_dict(
            {
                "negative_columns":
                    preprocessor.negative_columns,
                "dropped_columns":
                    preprocessor.columns_to_drop
            },
            "preprocessing_config.json"
        )

        mlflow.log_artifact(
            str(PROJECT_ROOT / "src/preprocessing.py"),
            artifact_path="preprocessing"
        )

        signature = infer_signature(
            X_train_clean.head(5),
            pipeline.predict_proba(
                X_train_clean.head(5)
            )
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=MODEL_NAME,
            signature=signature,
            input_example=X_train_clean.head(5),
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "src.preprocessing.CICIDSPreprocessor"
            ]
        )

    print("\nTraining completed.")

    print("\nValidation metrics:")
    print(val_metrics)

    print("\nTest metrics:")
    print(test_metrics)

    print("\nMLflow database:", MLFLOW_DB)
    print("MLflow artifacts:", MLFLOW_ARTIFACTS)


if __name__ == "__main__":
    main()
