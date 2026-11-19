import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


class ThreatDetector:

    def __init__(self):

        self.experiment_name = (
            "cyber-threat-detection-random-forest-model"
        )

        self.run_name = "random-forest-fine-tuned"

        self.model_name = "random_forest_model"

        # Configure tracking before creating the client
        mlflow.set_tracking_uri(
            f"sqlite:///{MLFLOW_DB}"
        )

        self.client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(
            self.experiment_name
        )

        if experiment is None:
            raise ValueError(
                "MLflow experiment not found."
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                f"attributes.run_name = '{self.run_name}' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=["start_time DESC"]
        )

        if runs.empty:
            raise ValueError(
                "No successful MLflow run found."
            )

        self.run_id = runs.iloc[0]["run_id"]

        run = self.client.get_run(self.run_id)

        params = run.data.params

        self.threshold = float(
            params["decision_threshold"]
        )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "Invalid decision threshold."
            )

        model_uri = (
            f"runs:/{self.run_id}/{self.model_name}"
        )

        # Load complete preprocessing + classifier pipeline
        self.model = mlflow.sklearn.load_model(
            model_uri
        )


        classifier = self.model.named_steps["classifier"]

        if 1 not in classifier.classes_:
            raise ValueError("Attack class (1) not found in model classes.")
        
        self.attack_class_index = list(classifier.classes_).index(1)

        


        

    def predict(self, features):

        probabilities = self.model.predict_proba(
            features
        )

    
        attack_probabilities = probabilities[
            :, self.attack_class_index
        ]

        predictions = (
            attack_probabilities >= self.threshold
        ).astype(int)

        labels = [
            "Attack" if prediction == 1 else "Benign"
            for prediction in predictions
        ]


        return pd.DataFrame(
            {
                "prediction": predictions,
                "label": labels,
                "attack_probability": attack_probabilities
            },
            index=features.index
        )

