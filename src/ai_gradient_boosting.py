#!/usr/bin/env python3
"""Model ceny za metr kwadratowy ofert sprzedaży mieszkań (15 miast, 2023-08 ... 2024-06).

Użycie:
    python model_haiku.py <katalog_z_plikami_csv> <katalog_wyjściowy>

Skrypt:
  1. wczytuje pliki apartments_pl_RRRR_MM.csv i łączy je (miesiąc bierze z nazwy pliku),
  2. czyści dane (duplikaty, wartości niemożliwe, skrajne ceny za m2),
  3. buduje cechy i model HistGradientBoostingRegressor w Pipeline,
  4. ocenia model 5-krotną walidacją krzyżową (R2, RMSE, MAE),
  5. zapisuje predykcje z walidacji do <katalog_wyjściowy>/predictions.csv.

Celem modelu jest cena za m2 (price / squareMeters).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
FILE_RE = re.compile(r"apartments_pl_(\d{4})_(\d{2})\.csv$")

# Kolumny liczbowe używane jako cechy (buildYear zastąpiony wiekiem budynku).
NUM_COLS = [
    "squareMeters", "rooms", "floor", "floorCount", "age",
    "latitude", "longitude", "centreDistance", "poiCount",
    "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance",
]
DISTANCE_COLS = [c for c in NUM_COLS if c.endswith("Distance")]
# Kolumny liczbowe w surowym pliku (przed czyszczeniem).
RAW_NUMERIC = [c for c in NUM_COLS if c != "age"] + ["buildYear", "price"]
BOOL_COLS = ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]
CAT_COLS = ["city", "type", "ownership", "buildingMaterial", "condition"]

BOOL_MAP = {"true": 1, "1": 1, "1.0": 1, "yes": 1,
            "false": 0, "0": 0, "0.0": 0, "no": 0}


def load_all(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("apartments_pl_*.csv")):
        match = FILE_RE.search(path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        df["month"] = f"{match.group(1)}-{match.group(2)}"
        frames.append(df)
    if not frames:
        sys.exit(f"Brak plików apartments_pl_RRRR_MM.csv w katalogu: {data_dir}")
    raw = pd.concat(frames, ignore_index=True)
    print(f"Wczytano {len(frames)} plików, łącznie {len(raw)} wierszy.")
    return raw


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates(subset=["id", "month"]).copy()

    for col in RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BOOL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower().map(BOOL_MAP)
    for col in CAT_COLS:
        df[col] = (df[col].astype(str).str.strip().str.lower()
                   .replace({"nan": "unknown", "": "unknown"}))

    # Odrzucenie rekordów z oczywiście błędnymi wartościami podstawowymi.
    mask = (
        df["price"].gt(0)
        & df["squareMeters"].between(10, 500)
        & (df["rooms"].isna() | df["rooms"].between(1, 10))
    )
    df = df[mask].copy()

    # Target: cena za m2 i obcięcie skrajnych 0,5% z każdej strony.
    df["price_per_m2"] = df["price"] / df["squareMeters"]
    lo, hi = df["price_per_m2"].quantile([0.005, 0.995])
    df = df[df["price_per_m2"].between(lo, hi)].copy()

    # Wartości niemożliwe zamieniamy na NaN (imputowane w pipeline).
    year = df["month"].str[:4].astype(int)
    df["buildYear"] = df["buildYear"].where(df["buildYear"].between(1800, 2026))
    df["age"] = (year - df["buildYear"]).where(lambda s: s.between(0, 250))
    df["floor"] = df["floor"].where(df["floor"].le(df["floorCount"]) | df["floorCount"].isna())
    for col in DISTANCE_COLS + ["poiCount"]:
        df[col] = df[col].where(df[col].ge(0))
    df["latitude"] = df["latitude"].where(df["latitude"].between(48.5, 55.5))
    df["longitude"] = df["longitude"].where(df["longitude"].between(13.5, 24.5))

    print(f"Po czyszczeniu: {len(df)} wierszy.")
    return df


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), NUM_COLS),
            ("bool", SimpleImputer(strategy="most_frequent"), BOOL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                  sparse_output=False), CAT_COLS),
        ]
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocess), ("model", regressor)])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("Użycie: python model_haiku.py <katalog_z_plikami_csv> <katalog_wyjściowy>")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    raw = load_all(data_dir)
    df = clean(raw)

    X = df[NUM_COLS + BOOL_COLS + CAT_COLS]
    y = df["price_per_m2"]

    model = build_model()
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)

    print("\nWalidacja krzyżowa 5-krotna (predykcje out-of-fold, cena za m2 w PLN):")
    print(f"  R2:   {r2_score(y, pred):.4f}")
    print(f"  RMSE: {rmse(y, pred):.2f}")
    print(f"  MAE:  {mean_absolute_error(y, pred):.2f}")

    print("\nWyniki w poszczególnych foldach:")
    for i, (_, test_idx) in enumerate(kf.split(X), 1):
        y_fold = y.iloc[test_idx]
        p_fold = pred[test_idx]
        print(f"  fold {i}: R2={r2_score(y_fold, p_fold):.4f} "
              f"RMSE={rmse(y_fold, p_fold):.2f} "
              f"MAE={mean_absolute_error(y_fold, p_fold):.2f}")

    out = pd.DataFrame({
        "id": df["id"].values,
        "month": df["month"].values,
        "actual": y.round(2).values,
        "predicted": np.round(pred, 2),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nZapisano predykcje: {out_path} ({len(out)} wierszy).")


if __name__ == "__main__":
    main()
