import pandas as pd
import os


def test_raw_data_exists():
    assert os.path.exists("data/raw/house_data.csv")


def test_raw_data_not_empty():
    df = pd.read_csv("data/raw/house_data.csv")

    assert len(df) > 0
    assert len(df.columns) > 0


def test_price_column_exists():
    df = pd.read_csv("data/raw/house_data.csv")

    assert "price" in df.columns