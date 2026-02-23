import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Chemins
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Chargement des données
X_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

# Chargement du modèle
with open(MODELS_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

print("Modèle chargé avec succès !")

# Prédictions
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # transformation inverse

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# Sauvegarde des métriques dans scores.json
scores = {
    "mse": round(mse, 4),
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "r2": round(r2, 4)
}

with open(METRICS_DIR / "scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(f"Métriques sauvegardées dans : {METRICS_DIR / 'scores.json'}")

# Sauvegarde des prédictions
predictions = pd.DataFrame({
    "y_test": y_test.values,
    "y_pred": y_pred
})

predictions.to_csv(PROCESSED_DIR / "predictions.csv", index=False)
print(f"Prédictions sauvegardées dans : {PROCESSED_DIR / 'predictions.csv'}")