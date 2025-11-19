
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

import joblib

DATA_PATH = Path("data/housing.csv")
MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

TARGET = "median_house_value"
CAT_COLS = ["ocean_proximity"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + 1e-9)
    df["bedrooms_per_room"]   = df["total_bedrooms"] / (df["total_rooms"] + 1e-9)
    df["population_per_household"] = df["population"] / (df["households"] + 1e-9)
    return df

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return add_engineered_features(df)

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
   
    cat_cols = [c for c in CAT_COLS if c in X.columns]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    r2: float
    pipeline: Pipeline

def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def available_models() -> Dict[str, object]:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, random_state=42),
    }
    if XGB_OK:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="reg:squarederror"
        )
    return models

def evaluate_model(name: str, estimator, X_train, X_test, y_train, y_test) -> ModelResult:
    preprocessor = make_preprocessor(X_train)
    pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2   = r2_score(y_test, preds)
    return ModelResult(name, mae, rmse, r2, pipe)

def compare_models(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X_train, X_test, y_train, y_test = split(df, test_size, seed)
    results = []
    best: Tuple[float, ModelResult] = (float("inf"), None)

    for name, est in available_models().items():
        res = evaluate_model(name, est, X_train, X_test, y_train, y_test)
        results.append(res)
        if res.rmse < best[0]:
            best = (res.rmse, res)

    return results, best[1]

def save_model(model: Pipeline, name: str) -> Path:
    path = MODEL_DIR / f"{name.replace(' ', '_').lower()}_pipeline.joblib"
    joblib.dump(model, path)
    return path

def load_model(path: Path) -> Pipeline:
    return joblib.load(path)
def save_model_with_meta(result: ModelResult, test_size: float, seed: int) -> Path:
    """
    Sauvegarde le modèle entraîné avec ses métriques et ses paramètres dans le dossier artifacts/.
    """
    payload = {
        "name": result.name,
        "metrics": {
            "mae": result.mae,
            "rmse": result.rmse,
            "r2": result.r2,
        },
        "params": {
            "test_size": test_size,
            "seed": seed,
        },
        "pipeline": result.pipeline,
    }

    # Création du nom de fichier (ex: random_forest_42_020.joblib)
    fname = f"{result.name.replace(' ', '_').lower()}_{seed}_{str(test_size).replace('.', '')}.joblib"
    path = MODEL_DIR / fname

    joblib.dump(payload, path)
    return path

def load_saved_payload(path: Path):
    """
    Charge le dict sauvegardé par save_model_with_meta(...).
    Retourne un dict avec les clés: name, metrics, params, pipeline.
    """
    return joblib.load(path)

# (optionnel) alias si tu préfères un autre nom d’import
load_model_payload = load_saved_payload