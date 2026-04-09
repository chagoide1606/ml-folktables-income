import joblib
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from preprocess import preprocess_data
from config import RANDOM_STATE, TEST_SIZE

data_source = ACSDataSource(
    survey_year='2021',
    horizon='1-Year',
    survey='person'
)

acs_data = data_source.get_data(states=["CA"], download=True)

acs_data = acs_data.copy()

features = ACSIncome.features

available_features = [f for f in features if f in acs_data.columns]

print("Features usadas:", available_features)

X = acs_data[available_features].to_numpy()

# target (renda > 50k)
y = (acs_data["PINCP"] > 50000).astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_test, preprocessors = preprocess_data(X_train, X_test)
import numpy as np
print("Tem NaN no treino?", np.isnan(X_train).any())

joblib.dump(preprocessors, "models/preprocessors.pkl")
print("Antes do SMOTE:", np.bincount(y_train))

smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Depois do SMOTE:", np.bincount(y_train))

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("F1-score: ", f1_score(y_test, y_pred))

joblib.dump(model, "models/model.pkl")
joblib.dump(preprocessors, "models/scaler.pkl")

print("Modelo treinado e salvo!")