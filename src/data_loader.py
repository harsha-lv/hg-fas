import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed.cleveland.data")

COLUMNS = [
    "age",       # Age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal, 4: asymptomatic)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol in mg/dl
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting electrocardiographic results (0, 1, 2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of peak exercise ST segment
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal",      # 3 = normal; 6 = fixed defect; 7 = reversible defect
    "target"     # Diagnosis of heart disease (0: <50% diameter narrowing, 1-4: >50% diameter narrowing)
]

FEATURE_NAMES = COLUMNS[:-1]

def download_and_load_data(cache_path=LOCAL_DATA_PATH):
    """
    Downloads the Cleveland Heart Disease dataset if not cached locally,
    cleans missing values denoted by '?', binarizes the target,
    and performs median imputation.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if not os.path.exists(cache_path):
        print(f"[DataLoader] Fetching UCI Cleveland dataset from {DATA_URL}...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp, open(cache_path, "wb") as f:
            f.write(resp.read())
        print(f"[DataLoader] Cached dataset to {cache_path}")
    else:
        print(f"[DataLoader] Loaded cached dataset from {cache_path}")
        
    df = pd.read_csv(cache_path, names=COLUMNS, na_values="?")
    
    # Binarize target: 0 = Healthy, 1-4 = Cardiovascular Disease Presence
    df["target"] = (df["target"] > 0).astype(int)
    
    # Missing value imputation (ca has 4 missing, thal has 2 missing)
    imputer = SimpleImputer(strategy="median")
    feature_matrix = imputer.fit_transform(df[FEATURE_NAMES])
    df_imputed = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    df_imputed["target"] = df["target"].values
    
    return df_imputed

def get_train_test_data(test_size=0.20, random_state=42):
    """
    Returns stratified train and test partitions.
    """
    df = download_and_load_data()
    X = df[FEATURE_NAMES].values
    y = df["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, FEATURE_NAMES, df
