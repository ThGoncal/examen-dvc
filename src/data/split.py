import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Chemins
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data" / "raw_data" / "raw.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed_data"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chargement des données
df = pd.read_csv(RAW_DATA_PATH)

# Feature engineering sur la date
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
df = df.drop(columns=["date"])

# Définition des features et de la cible
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sauvegarde des datasets
X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

print("Split terminé !")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Fichiers sauvegardés dans : {PROCESSED_DIR}")