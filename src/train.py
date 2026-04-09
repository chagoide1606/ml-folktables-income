import joblib
import numpy as np
from pathlib import Path

from folktables import ACSDataSource, ACSIncome
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importando do seu config local
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

# --- 1) Tratamento de dados faltantes diferenciado ---
# Definindo quais colunas são o quê dentro de available_features
num_cols = ["AGEP", "WKHP"]
cat_cols = ["COW", "SCHL", "MAR", "OCCP", "POBP", "SEX", "RAC1P"]

# Pegamos os índices numéricos e categóricos para o ColumnTransformer
num_indices = [available_features.index(f) for f in num_cols if f in available_features]
cat_indices = [available_features.index(f) for f in cat_cols if f in available_features]

imputer = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_indices),
        ("cat", SimpleImputer(strategy="most_frequent"), cat_indices),
    ],
    remainder="passthrough" # Mantém colunas que não foram explicitadas acima
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print("Tem NaN no treino?", np.isnan(X_train).any())
print("Tem NaN no teste?", np.isnan(X_test).any())
print("Antes do SMOTENC:", np.bincount(y_train))

# --- 2) Balanceamento com SMOTENC ---
# Como o ColumnTransformer pode mudar a ordem das colunas, 
# o ideal é mapear os novos índices categóricos. 
# No nosso caso, o "num" vem antes do "cat".
new_categorical_indices = list(range(len(num_indices), len(num_indices) + len(cat_indices)))

print("Novos índices categóricos para SMOTENC:", new_categorical_indices)

smotenc = SMOTENC(
    categorical_features=new_categorical_indices,
    random_state=RANDOM_STATE
)

X_train, y_train = smotenc.fit_resample(X_train, y_train)
print("Depois do SMOTENC:", np.bincount(y_train))

# --- 3) Padronização ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvamos o imputer (ColumnTransformer) e o scaler
preprocessors = (imputer, scaler)

# --- 4) Modelo ---
model = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --- 5) Avaliação ---
y_pred = model.predict(X_test)
print("F1-score:", f1_score(y_test, y_pred))

# --- 6) Salvar artefatos ---
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(preprocessors, MODEL_DIR / "preprocessors.pkl")

print("Modelo treinado e salvo com imputação otimizada!")