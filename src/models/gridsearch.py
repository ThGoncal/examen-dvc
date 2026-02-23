import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

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

# Grille des hyperparamètres
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0]
}

# GridSearchCV
grid_search = GridSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train_log)

# Meilleurs paramètres
best_params = grid_search.best_params_
print(f"Meilleurs paramètres : {best_params}")
print(f"Meilleur R² CV : {grid_search.best_score_:.4f}")

# Sauvegarde des meilleurs paramètres
with open(MODELS_DIR / "best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

print(f"Meilleurs paramètres sauvegardés dans : {MODELS_DIR / 'best_params.pkl'}")