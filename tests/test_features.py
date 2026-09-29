import pandas as pd
import os


def test_featured_data_exists():
    assert os.path.exists(
        "data/processed/featured_house_data.csv"
    )


def test_preprocessor_exists():
    assert os.path.exists(
        "models/trained/preprocessor.pkl"
    )


def test_engineered_columns_exist():

    df = pd.read_csv(
        "data/processed/featured_house_data.csv"
    )

    assert len(df.columns) > 1

    assert "price" in df.columns
