import argparse
import os
import joblib
import yaml
import logging
import platform

import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
import sklearn


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Arguments
# ---------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to model_config.yaml"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to processed dataset"
    )

    parser.add_argument(
        "--models-dir",
        required=True,
        help="Directory to save models"
    )

    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow Tracking URI"
    )

    return parser.parse_args()


# ---------------------------------------------------
# Model Factory
# ---------------------------------------------------

def get_model_instance(name, params):

    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor
    }

    if name not in model_map:
        raise ValueError(
            f"Unsupported model: {name}"
        )

    return model_map**params


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(
            args.mlflow_tracking_uri
        )

    mlflow.set_experiment(
        model_cfg["name"]
    )

    data = pd.read_csv(args.data)

    target = model_cfg["target_variable"]

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    best_model = None
    best_model_name = None
    best_r2 = float("-inf")
    best_mae = None

    client = MlflowClient()

    logger.info("Starting model comparison")

    for model_name, params in model_cfg["candidate_models"].items():

        model = get_model_instance(
            model_name,
            params
        )

        with mlflow.start_run(run_name=model_name):

            logger.info(
                f"Training model: {model_name}"
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            mae = float(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            )

            r2 = float(
                r2_score(
                    y_test,
                    predictions
                )
            )

            mlflow.log_params(params)

            mlflow.log_metric(
                "mae",
                mae
            )

            mlflow.log_metric(
                "r2",
                r2
            )

            mlflow.sklearn.log_model(
                model,
                name=model_name
            )

            logger.info(
                f"{model_name} -> "
                f"MAE={mae:.2f} "
                f"R2={r2:.4f}"
            )

            if r2 > best_r2:

                best_r2 = r2
                best_mae = mae

                best_model = model
                best_model_name = model_name

    logger.info(
        f"Best Model: {best_model_name}"
    )

    logger.info(
        f"Best R2: {best_r2:.4f}"
    )

    # -----------------------------------------
    # Save Best Model
    # -----------------------------------------

    trained_dir = os.path.join(
        args.models_dir,
        "trained"
    )

    os.makedirs(
        trained_dir,
        exist_ok=True
    )

    model_name = model_cfg["name"]

    save_path = os.path.join(
        trained_dir,
        f"{model_name}.pkl"
    )

    joblib.dump(
        best_model,
        save_path
    )

    logger.info(
        f"Saved best model to: {save_path}"
    )

    # -----------------------------------------
    # Register Best Model
    # -----------------------------------------

    with mlflow.start_run(
        run_name="best_model_registration"
    ):

        mlflow.log_param(
            "selected_model",
            best_model_name
        )

        mlflow.log_metric(
            "best_r2",
            best_r2
        )

        mlflow.log_metric(
            "best_mae",
            best_mae
        )

        model_info = mlflow.sklearn.log_model(
            best_model,
            name="best_model"
        )

        logger.info(
            "Registering best model to MLflow Registry"
        )

        try:

            client.create_registered_model(
                model_name
            )

            logger.info(
                f"Created registered model: "
                f"{model_name}"
            )

        except Exception:

            logger.info(
                f"{model_name} already exists"
            )

        model_version = client.create_model_version(
            name=model_name,
            source=model_info.model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Optional Stage Transition

        try:

            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )

        except Exception as ex:

            logger.warning(
                f"Stage transition skipped: {ex}"
            )

        description = f"""
Model Name: {model_name}

Best Algorithm: {best_model_name}

MAE: {best_mae:.2f}

R2 Score: {best_r2:.4f}

Training Dataset: {args.data}
"""

        client.update_registered_model(
            name=model_name,
            description=description
        )

        client.set_registered_model_tag(
            model_name,
            "best_algorithm",
            best_model_name
        )

        client.set_registered_model_tag(
            model_name,
            "python_version",
            platform.python_version()
        )

        client.set_registered_model_tag(
            model_name,
            "scikit_learn_version",
            sklearn.__version__
        )

        client.set_registered_model_tag(
            model_name,
            "xgboost_version",
            xgb.__version__
        )

        client.set_registered_model_tag(
            model_name,
            "pandas_version",
            pd.__version__
        )

        client.set_registered_model_tag(
            model_name,
            "numpy_version",
            np.__version__
        )

    logger.info("Training completed successfully")


if __name__ == "__main__":

    args = parse_args()

    main(args)