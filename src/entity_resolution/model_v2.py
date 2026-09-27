"""V2 ML model training and probability prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(
    X_train: pd.DataFrame,
    y_train: Iterable[int],
    *,
    random_state: int = 42,
    model_type: str = "logistic",
):
    """Train the selected V2 classifier."""
    y = np.asarray(list(y_train), dtype=int)

    if model_type.lower() == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    model.fit(X_train, y)
    return model


def predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities."""
    return model.predict_proba(X)[:, 1]


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)