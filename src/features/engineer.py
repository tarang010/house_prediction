
import argparse
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("feature-engineering")


def create_features(df):
    """Create prediction-time features without using the target as an input."""
    df = df.copy()

    # Normalize column names for consistent processing.
    df.columns = [c.strip().lower() for c in df.columns]

    if "price" not in df.columns:
        raise ValueError("Required target column 'price' is missing.")

    # Remove identifiers that are not predictive features.
    df.drop(
        columns=["source_prediction_id", "prediction_id"],
        errors="ignore",
        inplace=True,
    )

    # Ensure target and numeric source columns are numeric.
    for col in ["price", "sqft", "bedrooms", "bathrooms", "year_built"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["price"].isna().any():
        logger.warning("Rows with missing/non-numeric target prices will be removed.")
        df = df.dropna(subset=["price"]).copy()

    if (df["sqft"].dropna() <= 0).any():
        logger.warning("Non-positive sqft values found; check data quality.")

    current_year = datetime.now().year
    df["house_age"] = current_year - df["year_built"]

    # Avoid division-by-zero and invalid ratios.
    df["bed_bath_ratio"] = (
        df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Missing ratios are handled by the numerical imputer.
    logger.info("Created house_age and bed_bath_ratio features.")
    return df.reset_index(drop=True)


def create_preprocessor():
    """Build preprocessing that will be fitted only on training data."""
    numeric_features = [
        "sqft",
        "bedrooms",
        "bathrooms",
        "year_built",
        "house_age",
        "bed_bath_ratio",
    ]
    categorical_features = ["location", "condition"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def run_feature_engineering(input_file, output_file, preprocessor_file):
    logger.info("Loading data from %s", input_file)
    df = pd.read_csv(input_file)
    df_featured = create_features(df)

    # Validate required input features.
    required = [
        "price", "sqft", "bedrooms", "bathrooms",
        "year_built", "location", "condition",
    ]
    missing = [col for col in required if col not in df_featured.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Save raw engineered features. The trainer fits the preprocessor
    # on its training split to prevent preprocessing leakage.
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_file) or ".", exist_ok=True)

    df_featured.to_csv(output_file, index=False)
    joblib.dump(create_preprocessor(), preprocessor_file)

    logger.info("Saved engineered dataset: %s", df_featured.shape)
    logger.info("Saved preprocessor template to %s", preprocessor_file)
    logger.info("Saved data to %s", output_file)
    return df_featured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Engineer house-price training features."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preprocessor", required=True)
    args = parser.parse_args()

    run_feature_engineering(
        args.input, args.output, args.preprocessor
    )
