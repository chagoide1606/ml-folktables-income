import joblib
import numpy as np
from pathlib import Path

from folktables import ACSDataSource, ACSIncome
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, TEST_SIZE


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


data_source = ACSDataSource(
    survey_year="2021",
    horizon="1-Year",
    survey="person"
)

acs_data = data_source.get_data(states=["CA"], download=True).copy()

features = ACSIncome.features
available_features = [f for f in features if f in acs_data.columns]

print("Features usadas:", available_features)

X = acs_data[available_features].to_numpy()
y = (acs_data["PINCP"] > 50000).astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# 1) Tratamento de dados faltantes
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("Tem NaN no treino?", np.isnan(X_train).any())
print("Tem NaN no teste?", np.isnan(X_test).any())

print("Antes do SMOTENC:", np.bincount(y_train))

# 2) Balanceamento com SMOTENC
# Índices das colunas categóricas dentro de available_features
categorical_feature_names = {"COW", "SCHL", "MAR", "OCCP", "POBP", "SEX", "RAC1P"}
categorical_indices = [
    i for i, feature_name in enumerate(available_features)
    if feature_name in categorical_feature_names
]

print("Índices categóricos para SMOTENC:", categorical_indices)

smotenc = SMOTENC(
    categorical_features=categorical_indices,
    random_state=RANDOM_STATE
)

X_train, y_train = smotenc.fit_resample(X_train, y_train)

print("Depois do SMOTENC:", np.bincount(y_train))

# 3) Padronização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

preprocessors = (imputer, scaler)

# 4) Modelo
model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5) Avaliação
y_pred = model.predict(X_test)
print("F1-score:", f1_score(y_test, y_pred))

# 6) Salvar artefatos
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(preprocessors, MODEL_DIR / "preprocessors.pkl")

print("Modelo treinado e salvo!")