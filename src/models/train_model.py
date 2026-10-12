
import argparse
import json
import logging
import os
import platform

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from .features.engineer import create_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def get_model_instance(name, params):
    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    target = model_cfg["target_variable"]
    model_name = model_cfg["name"]
    algorithm = model_cfg["best_model"]
    params = model_cfg.get("parameters", {})

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(model_name)

    data = pd.read_csv(args.data)

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found.")

    # Never use the target or identifiers as input features.
    excluded = [
        target,
        "source_prediction_id",
        "prediction_id",
    ]
    X = data.drop(columns=excluded, errors="ignore")
    y = pd.to_numeric(data[target], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    if len(X) < 10:
        raise ValueError("Not enough valid rows to train the model.")

    # Split BEFORE fitting preprocessing to prevent leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", get_model_instance(algorithm, params)),
    ])

    os.makedirs(args.models_dir, exist_ok=True)
    trained_dir = os.path.join(args.models_dir, "trained")
    os.makedirs(trained_dir, exist_ok=True)

    with mlflow.start_run(run_name=algorithm) as run:
        logger.info("Training %s on %d rows", algorithm, len(X_train))

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(root_mean_squared_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        mlflow.log_params(params)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("target_variable", target)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

        # Log the entire preprocessing + estimator pipeline.
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
            logger.info("Created registered model %s", model_name)
        except mlflow.exceptions.MlflowException:
            logger.info("Registered model %s already exists", model_name)

        version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id,
        )

        client.update_model_version(
            name=model_name,
            version=version.version,
            description=(
                f"Algorithm: {algorithm}; target: {target}; "
                f"MAE: {mae:.2f}; RMSE: {rmse:.2f}; R2: {r2:.4f}"
            ),
        )

        # Retain the existing Staging workflow.
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging",
        )

        tags = {
            "algorithm": algorithm,
            "target_variable": target,
            "training_dataset": args.data,
            "mae": str(mae),
            "rmse": str(rmse),
            "r2": str(r2),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }

        for key, value in tags.items():
            client.set_model_version_tag(
                model_name, version.version, key, value
            )

        # Save the complete fitted pipeline, not just XGBoost.
        save_path = os.path.join(trained_dir, f"{model_name}.pkl")
        joblib.dump(pipeline, save_path)

        metrics = {
            "model": algorithm,
            "model_name": model_name,
            "model_version": str(version.version),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "dataset": args.data,
            "model_path": save_path,
            "mlflow_run_id": run.info.run_id,
        }

        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Model version: %s", version.version)
        logger.info("Model saved: %s", save_path)
        logger.info("MAE: %.2f | RMSE: %.2f | R2: %.4f", mae, rmse, r2)


if __name__ == "__main__":
    main(parse_args())
