import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "model.pkl")
preprocessors = joblib.load(MODEL_DIR / "preprocessors.pkl")
imputer, scaler = preprocessors


def predict_income(data):
    features = np.array([[
        data["AGEP"],
        data["COW"],
        data["SCHL"],
        data["MAR"],
        data["OCCP"],
        data["POBP"],
        data["WKHP"],
        data["SEX"],
        data["RAC1P"]
    ]], dtype=float)

    features = imputer.transform(features)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = None

    return int(prediction), probability