import os
import joblib


MODEL_PATH = "models/trained/HousePricePredictor.pkl"


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_has_predict_method():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")
