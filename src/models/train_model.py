import argparse
import json
import logging
import os
import platform
import sys

# --------------------------------------------------
# FIX: Add project root to Python's import path.
# This allows Jenkins to import src.features.engineer
# when executing src/models/train_model.py directly.
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.features.engineer import create_preprocessor


# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and register a house price model."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--mlflow-tracking-uri", default=None)

    return parser.parse_args()


# --------------------------------------------------
# MODEL FACTORY
# --------------------------------------------------
def get_model_instance(name, params):
    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor,
    }

    if name not in model_map:
        raise ValueError(
            f"Unsupported model: {name}. "
            f"Supported models: {', '.join(model_map.keys())}"
        )

    return model_map[name](**params)


# --------------------------------------------------
# TRAINING PIPELINE
# --------------------------------------------------
def main(args):
    # Load configuration.
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_cfg = config["model"]

    target = model_cfg["target_variable"]
    model_name = model_cfg["name"]
    algorithm = model_cfg["best_model"]
    params = model_cfg.get("parameters", {})

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Model name: %s", model_name)
    logger.info("Algorithm: %s", algorithm)
    logger.info("Target variable: %s", target)

    # Configure MLflow.
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(model_name)

    # Load dataset.
    data_path = os.path.abspath(args.data)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Training dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    # Normalize column names.
    data.columns = [str(column).strip().lower() for column in data.columns]
    target = target.strip().lower()

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    # Remove target and identifiers from model inputs.
    # Identifiers must not be used as predictive features.
    excluded = [
        target,
        "source_prediction_id",
        "prediction_id",
    ]

    X = data.drop(columns=excluded, errors="ignore").copy()
    y = pd.to_numeric(data[target], errors="coerce")

    # Remove rows with missing or non-numeric target values.
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Fail early if the dataset is not suitable.
    if len(X) < 10:
        raise ValueError(
            f"Not enough valid rows to train the model: {len(X)}"
        )

    if X.empty or len(X.columns) == 0:
        raise ValueError("No input features remain after exclusions.")

    logger.info("Dataset: %s", data_path)
    logger.info("Valid rows: %d", len(X))
    logger.info("Input columns: %s", list(X.columns))

    # Split before fitting preprocessing to avoid fitting the
    # imputers and encoders on the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    logger.info("Training rows: %d", len(X_train))
    logger.info("Test rows: %d", len(X_test))

    # The preprocessor handles numerical imputation and
    # categorical one-hot encoding inside the fitted pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", get_model_instance(algorithm, params)),
        ]
    )

    # Create output directories.
    models_dir = os.path.abspath(args.models_dir)
    trained_dir = os.path.join(models_dir, "trained")

    os.makedirs(trained_dir, exist_ok=True)

    # Train and log the run.
    with mlflow.start_run(run_name=algorithm) as run:
        logger.info("Starting model training...")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        logger.info(
            "Evaluation: MAE=%.2f | RMSE=%.2f | R2=%.4f",
            mae,
            rmse,
            r2,
        )

        # Log model parameters and evaluation metrics.
        mlflow.log_params(params)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("target_variable", target)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("dataset", data_path)

        mlflow.log_metrics(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        # Log the entire fitted pipeline, including preprocessing.
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
        )

        model_uri = f"runs:/{run.info.run_id}/model"

        # Register the trained model.
        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
            logger.info(
                "Created registered model: %s",
                model_name,
            )
        except mlflow.exceptions.MlflowException:
            # The registered model may already exist.
            logger.info(
                "Registered model already exists: %s",
                model_name,
            )

        version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id,
        )

        # Wait for the model version to become ready when supported.
        client.wait_until_ready(
            name=model_name,
            version=version.version,
            timeout=300,
        )

        client.update_model_version(
            name=model_name,
            version=version.version,
            description=(
                f"Algorithm: {algorithm}; "
                f"target: {target}; "
                f"MAE: {mae:.2f}; "
                f"RMSE: {rmse:.2f}; "
                f"R2: {r2:.4f}"
            ),
        )

        # Retain the existing Staging workflow.
        # Note: MLflow model stages are deprecated in newer versions.
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging",
        )

        # Record reproducibility metadata.
        tags = {
            "algorithm": algorithm,
            "target_variable": target,
            "training_dataset": data_path,
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
                name=model_name,
                version=version.version,
                key=key,
                value=value,
            )

        # Save the COMPLETE fitted pipeline, not only XGBoost.
        save_path = os.path.join(
            trained_dir,
            f"{model_name}.pkl",
        )

        joblib.dump(pipeline, save_path)

        # Save training results for downstream Jenkins stages.
        metrics = {
            "model": algorithm,
            "model_name": model_name,
            "model_version": str(version.version),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "dataset": data_path,
            "model_path": save_path,
            "mlflow_run_id": run.info.run_id,
        }

        metrics_path = os.path.join(PROJECT_ROOT, "metrics.json")

        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        logger.info("Model version: %s", version.version)
        logger.info("Model saved: %s", save_path)
        logger.info("Metrics saved: %s", metrics_path)
        logger.info(
            "Training completed successfully. "
            "MAE: %.2f | RMSE: %.2f | R2: %.4f",
            mae,
            rmse,
            r2,
        )


if __name__ == "__main__":
    main(parse_args())
