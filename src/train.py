import joblib
import numpy as np
from pathlib import Path

from folktables import ACSDataSource, ACSIncome
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assumindo que esses valores estão no seu config.py
try:
    from config import RANDOM_STATE, TEST_SIZE
except ImportError:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1) Download e Correção de Esquema (2023)
data_source = ACSDataSource(
    survey_year="2023",
    horizon="1-Year",
    survey="person"
)

print("Baixando dados de 2023...")
acs_data = data_source.get_data(states=["CA"], download=True).copy()

# Mapeamento essencial para dados pós-2021
# RELP virou RELSHIPP nas versões recentes do Censo
if "RELSHIPP" in acs_data.columns and "RELP" not in acs_data.columns:
    print("Corrigindo coluna RELSHIPP -> RELP para compatibilidade.")
    acs_data = acs_data.rename(columns={"RELSHIPP": "RELP"})

# Usar as features padrão do ACSIncome (agora que corrigimos o RELP)
features_to_use = ACSIncome.features
X = acs_data[features_to_use].to_numpy()
y = (acs_data["PINCP"] > 50000).astype(int).to_numpy()

print(f"Features utilizadas: {features_to_use}")

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

imputer = SimpleImputer(strategy="most_frequent")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


categorical_feature_names = {"COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"}
categorical_indices = [
    i for i, f_name in enumerate(features_to_use)
    if f_name in categorical_feature_names
]

print(f"Distribuição original: {np.bincount(y_train)}")
print(f"Aplicando SMOTENC nos índices: {categorical_indices}")

smotenc = SMOTENC(
    categorical_features=categorical_indices,
    random_state=RANDOM_STATE,
)

X_train, y_train = smotenc.fit_resample(X_train, y_train)
print(f"Distribuição pós-SMOTENC: {np.bincount(y_train)}")

# 5) Padronização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6) Treinamento do Modelo
model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    max_depth=20, # Adicionado para evitar overfitting excessivo em dados ruidosos
    n_jobs=-1
)

print("Treinando o Random Forest...")
model.fit(X_train, y_train)

# 7) Avaliação detalhada
y_pred = model.predict(X_test)
print("\n--- Relatório de Performance ---")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8) Salvar artefatos
joblib.dump(model, MODEL_DIR / "model.pkl")
# Guardamos o imputer e o scaler juntos para facilitar o pipeline de inferência
joblib.dump({"imputer": imputer, "scaler": scaler, "features": features_to_use}, MODEL_DIR / "preprocessors.pkl")

print(f"Modelo e pré-processadores salvos em: {MODEL_DIR}")