import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor

# Chemins
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Chargement des données
X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

# Transformation logarithmique de la cible
y_train_log = np.log1p(y_train)

# Chargement des meilleurs paramètres
with open(MODELS_DIR / "best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

print(f"Paramètres chargés : {best_params}")

# Entraînement du modèle avec les meilleurs paramètres
model = XGBRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_log)

# Sauvegarde du modèle entraîné
with open(MODELS_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Modèle entraîné et sauvegardé dans : {MODELS_DIR / 'model.pkl'}")