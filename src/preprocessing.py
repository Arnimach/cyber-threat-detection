
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CICIDSPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.negative_columns = [
            'Flow IAT Min',
            'Flow Duration',
            'Flow Packets/s',
            'Flow IAT Mean',
            'Flow IAT Max',
            'Fwd Header Length',
            'min_seg_size_forward',
            'Flow Bytes/s',
            'Bwd Header Length',
            'Fwd IAT Min'
        ]

        self.columns_to_drop = [
            "Fwd Header Length.1",
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
            "CWE Flag Count",
            "Fwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
            "Subflow Fwd Packets",
            "Subflow Bwd Packets",
            "Subflow Fwd Bytes",
            "Fwd PSH Flags",
        ]

        self.feature_names_ = None

    def fit(self, X, y=None):

        X = X.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        self.feature_names_ = X.columns.tolist()

        return self

    def transform(self, X):

        X = X.copy()

        X = X.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        if self.feature_names_ is None:
            raise ValueError("Preprocessor has not been fitted.")

        # Check required features
        missing = set(self.feature_names_) - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )

        # Ignore extra features and preserve training order
        X = X.loc[:, self.feature_names_]

        # Reject missing values
        if X.isna().any().any():
            raise ValueError(
                "Input contains missing values (NaN)."
            )

        # Reject infinite values
        values = X.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Input contains infinite values."
            )

        return X
