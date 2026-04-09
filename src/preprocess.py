from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(X_train, X_test):
    imputer = SimpleImputer(strategy="median")
    
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, (imputer, scaler)