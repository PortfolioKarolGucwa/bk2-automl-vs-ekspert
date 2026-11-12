#!/usr/bin/env python3
"""Random Forest przewidujący cenę za metr kwadratowy mieszkań.

Użycie:
    python model_rf.py <katalog_z_csv> <katalog_wyjsciowy>

Dane: pliki apartments_pl_RRRR_MM.csv (jeden plik na miesiąc).
Wynik: <katalog_wyjsciowy>/predictions.csv (id, month, actual, predicted).
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET = "pricePerSqM"
ID_COL = "id"
MONTH_COL = "month"
FILE_PATTERN = "apartments_pl_*.csv"
FILE_RE = re.compile(r"apartments_pl_(\d{4})_(\d{2})\.csv$")
BASE_MONTH_INDEX = 2023 * 12 + 8  # 2023-08 jako punkt zerowy

NUMERIC_COLS = [
    "squareMeters", "rooms", "floor", "floorCount", "buildYear",
    "latitude", "longitude", "centreDistance", "poiCount",
    "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance",
]
BOOL_COLS = ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]
CATEGORICAL_COLS = ["city", "type", "ownership", "buildingMaterial", "condition"]

# Cecha pochodna: numer miesiąca od 2023-08 (trend cenowy w czasie).
NUMERIC_FEATURES = NUMERIC_COLS + BOOL_COLS + ["monthIndex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLS

BOOL_MAP = {
    "true": 1.0, "yes": 1.0, "y": 1.0, "tak": 1.0, "1": 1.0, "1.0": 1.0,
    "false": 0.0, "no": 0.0, "n": 0.0, "nie": 0.0, "0": 0.0, "0.0": 0.0,
}


def load_all(data_dir):
    """Wczytuje wszystkie pliki miesięczne i dokleja kolumnę month (RRRR-MM)."""
    paths = sorted(glob.glob(os.path.join(data_dir, FILE_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"Brak plików {FILE_PATTERN} w katalogu: {data_dir}")

    frames = []
    for path in paths:
        match = FILE_RE.search(os.path.basename(path))
        if match is None:
            print(f"Pominięto plik o nieoczekiwanej nazwie: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df[MONTH_COL] = f"{match.group(1)}-{match.group(2)}"
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nie znaleziono plików o poprawnej nazwie.")
    return pd.concat(frames, ignore_index=True)


def _to_bool(series):
    """Zamienia wartości logiczne (True/False, tak/nie, 1/0) na 0.0/1.0; reszta to NaN."""
    if series.dtype == bool:
        return series.astype(float)
    return series.astype(str).str.strip().str.lower().map(BOOL_MAP).astype(float)


def clean(raw):
    """Czyści dane i zwraca ramkę z cechami (FEATURES), identyfikatorami i kolumną TARGET."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset=[ID_COL, MONTH_COL])

    for col in NUMERIC_COLS + ["price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BOOL_COLS:
        df[col] = _to_bool(df[col])
    for col in CATEGORICAL_COLS:
        values = df[col].astype(str).str.strip()
        df[col] = values.where(~values.isin(["", "nan", "None"]), np.nan)

    # Usunięcie rekordów bez ceny lub powierzchni oraz oczywistych błędów.
    df = df.dropna(subset=["price", "squareMeters"])
    df = df[(df["price"] > 0) & df["squareMeters"].between(10, 300)].copy()

    # Wartości niemożliwe zamieniamy na NaN (imputowane w modelu).
    df.loc[~df["buildYear"].between(1800, 2030), "buildYear"] = np.nan
    df.loc[df["floor"] > df["floorCount"], "floor"] = np.nan

    # Cel: cena za metr kwadratowy. Cena całkowita nie trafia do cech (brak przecieku).
    df[TARGET] = df["price"] / df["squareMeters"]
    lo, hi = df[TARGET].quantile([0.005, 0.995])
    df = df[df[TARGET].between(lo, hi)].copy()

    year = df[MONTH_COL].str.slice(0, 4).astype(int)
    month = df[MONTH_COL].str.slice(5, 7).astype(int)
    df["monthIndex"] = year * 12 + month - BASE_MONTH_INDEX

    return df[[ID_COL, MONTH_COL] + FEATURES + [TARGET]].reset_index(drop=True)


def build_model():
    """Pipeline: imputacja + one-hot dla kategorii, następnie Random Forest (niedopasowany)."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])
    forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocessor), ("rf", forest)])


def _metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def evaluate(model, X, y):
    """5-krotna walidacja krzyżowa. Zwraca predykcje z walidacji i metryki per fold."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        estimator = clone(model)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = estimator.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        scores = _metrics(y.iloc[test_idx], pred)
        fold_rows.append(scores)
        print(f"Fold {fold}: R2={scores['R2']:.4f} RMSE={scores['RMSE']:.2f} MAE={scores['MAE']:.2f}")
    return oof, pd.DataFrame(fold_rows)


def main():
    if len(sys.argv) != 3:
        print("Użycie: python model_rf.py <katalog_z_csv> <katalog_wyjsciowy>", file=sys.stderr)
        sys.exit(1)
    data_dir, out_dir = sys.argv[1], sys.argv[2]

    raw = load_all(data_dir)
    print(f"Wczytano wierszy: {len(raw)}")

    df = clean(raw)
    print(f"Po czyszczeniu: {len(df)} wierszy, {len(FEATURES)} cech")

    X = df[FEATURES]
    y = df[TARGET]
    model = build_model()

    oof, fold_scores = evaluate(model, X, y)

    print("\nŚrednia z 5 foldów (±odchylenie standardowe):")
    for name in ["R2", "RMSE", "MAE"]:
        mean = fold_scores[name].mean()
        std = fold_scores[name].std()
        print(f"  {name}: {mean:.4f} (±{std:.4f})")

    pooled = _metrics(y, oof)
    print("\nMetryki na wszystkich predykcjach walidacyjnych:")
    for name, value in pooled.items():
        print(f"  {name}: {value:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    pred_df = pd.DataFrame({
        ID_COL: df[ID_COL],
        MONTH_COL: df[MONTH_COL],
        "actual": y,
        "predicted": oof,
    })
    out_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(out_path, index=False)
    print(f"\nZapisano predykcje: {out_path}")


if __name__ == "__main__":
    main()
