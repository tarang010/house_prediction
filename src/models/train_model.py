import argparse
import os
import pandas as pd
import joblib
import xgboost as xgb
import yaml
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train final model from config."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model_config.yaml"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed CSV dataset"
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory to save trained model"
    )

    return parser.parse_args()


# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    model_map = {
    "LinearRegression": LinearRegression,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "XGBoost": xgb.XGBRegressor
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
        return model_map**params
# -----------------------------
# Main Logic
# -----------------------------
def main(args):

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    # Load data
    logger.info(f"Loading data from {args.data}")

    data = pd.read_csv(args.data)

    target = model_cfg["target_variable"]

    # Features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = get_model_instance(
        model_cfg["best_model"],
        model_cfg["parameters"]
    )

    logger.info(
        f"Training model: {model_cfg['best_model']}"
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"R² Score: {r2:.4f}")

    # Create output directory
    trained_dir = os.path.join(
        args.models_dir,
        "trained"
    )

    os.makedirs(
        trained_dir,
        exist_ok=True
    )

    # Save model
    model_name = model_cfg["name"]

    save_path = os.path.join(
        trained_dir,
        f"{model_name}.pkl"
    )

    joblib.dump(model, save_path)

    logger.info(
        f"Model saved successfully: {save_path}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
