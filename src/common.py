"""Rzeczy wspólne dla wszystkich skryptów: wczytanie plików, cena za m2, zmienne BK2%, metryki, podział CV."""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

SEED = 42
N_EXPECTED = 195_568          # N z rozdziału
TARGET = "pricePerSqm"
MONTH = "monthIdx"            # 0 = VIII 2023 ... 10 = VI 2024
BK2 = ["bk2_active", "panic_month"]


def load_sales(raw_dir: str) -> pd.DataFrame:
    """Skleja 11 plików sprzedaży. Daty w danych nie ma, więc miesiąc biorę z nazwy pliku."""
    files = sorted(glob.glob(os.path.join(raw_dir, "apartments_pl_*.csv")))
    if len(files) != 11:
        raise FileNotFoundError(f"Oczekiwano 11 plików apartments_pl_*.csv, jest {len(files)}")
    frames = []
    for path in files:
        year, month = re.search(r"(\d{4})_(\d{2})\.csv$", path).groups()
        part = pd.read_csv(path)
        part["period"] = f"{year}-{month}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Czyszczenie jak w rozdziale: duplikaty wierszy i rekordy bez ceny lub powierzchni.

    W praktyce nic nie wypada. Te same oferty (to samo `id`) w kolejnych
    miesiącach zostawiam, bo inaczej N nie zgadzałoby się z rozdziałem.
    """
    out = df.drop_duplicates().dropna(subset=["price", "squareMeters"]).reset_index(drop=True)
    out[TARGET] = out["price"] / out["squareMeters"]
    return out


def add_time_features(df: pd.DataFrame, period_col: str = "period") -> pd.DataFrame:
    """Numer miesiąca i dwie zmienne o BK2%, których nie da się wyczytać z CSV.

    bk2_active  = 1 dla VIII-XII 2023 (program działał),
    panic_month = 1 dla XII 2023 (kończył się limit, ludzie kupowali na zapas).
    """
    out = df.copy()
    out[MONTH] = out[period_col].factorize(sort=True)[0]
    out["bk2_active"] = (out[period_col] <= "2023-12").astype(int)
    out["panic_month"] = (out[period_col] == "2023-12").astype(int)
    return out


def prepare(raw_dir: str) -> pd.DataFrame:
    df = add_time_features(clean(load_sales(raw_dir)))
    assert len(df) == N_EXPECTED, f"N = {len(df)}, w pracy {N_EXPECTED}"
    return df


def subperiods(period: pd.Series) -> dict[str, np.ndarray]:
    """Trzy podokresy z tab. X.4."""
    return {
        "VIII-XII 2023 (BK2% aktywny)": (period <= "2023-12").to_numpy(),
        "XII 2023 (panika zakupowa)": (period == "2023-12").to_numpy(),
        "I-VI 2024 (BK2% zawieszone)": (period >= "2024-01").to_numpy(),
    }


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"R2": r2(y, pred), "RMSE": rmse(y, pred), "MAE": mae(y, pred)}


def make_cv(scheme: str):
    """kfold: losowy podział na 5 części, tak jak w rozdziale.
    group: podział po `id`, żeby to samo mieszkanie nie trafiło naraz do treningu i testu."""
    if scheme == "kfold":
        return KFold(n_splits=5, shuffle=True, random_state=SEED)
    if scheme == "group":
        return GroupKFold(n_splits=5)
    raise ValueError(scheme)
