import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "model.pkl")
preprocessors = joblib.load(MODEL_DIR / "preprocessors.pkl")

# Compatibilidade com dois formatos:
# - legado: tupla (imputer, scaler)
# - atual: dict {"imputer": ..., "scaler": ..., "features"/"feature_columns": [...]}
if isinstance(preprocessors, dict):
    imputer = preprocessors["imputer"]
    scaler = preprocessors["scaler"]
    feature_columns = preprocessors.get("feature_columns") or preprocessors.get("features")
else:
    imputer, scaler = preprocessors
    feature_columns = None

if not feature_columns:
    feature_columns = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "WKHP", "SEX", "RAC1P"]


def predict_income(data):
    # Se o modelo treinou com features extras (ex.: RELP), preenche com 0 quando ausente no app.
    row = [data.get(col, 0) for col in feature_columns]
    features = np.array([row], dtype=float)

    features = imputer.transform(features)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = None

    return int(prediction), probability