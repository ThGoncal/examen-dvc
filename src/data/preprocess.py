import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Chemins
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"

# Chargement des datasets issus du split
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrame pour garder les noms de colonnes
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarde des datasets normalisés
X_train_scaled.to_csv(PROCESSED_DIR / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(PROCESSED_DIR / "X_test_scaled.csv", index=False)

# Sauvegarde du scaler
with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Preprocessing terminé !")
print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"Fichiers sauvegardés dans : {PROCESSED_DIR}")