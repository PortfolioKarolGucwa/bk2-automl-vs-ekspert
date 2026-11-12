"""Część opisowa: tab. X.1, tab. X.2, braki danych, korelacje.

Uruchomienie:  python src/descriptive.py data/raw results
"""
import os
import sys

import numpy as np
import pandas as pd

from common import SEED, TARGET, prepare

CITIES_X1 = ["warszawa", "krakow", "wroclaw", "gdansk", "poznan", "lodz"]


def table_x1(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["city"].isin(CITIES_X1)].groupby("city")
    out = pd.DataFrame({
        "liczba_obs": g.size(),
        "sr_cena_m2": g[TARGET].mean(),
        "min_cena": g["price"].min(),
        "max_cena": g["price"].max(),
        "sr_pow_m2": g["squareMeters"].mean(),
        "sr_odl_centrum_km": g["centreDistance"].mean(),
    })
    return out.loc[CITIES_X1]


def table_x2(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("period")
    out = pd.DataFrame({"sr_cena_m2": g[TARGET].mean(), "liczba_obs": g.size()})
    out["zmiana_mm_proc"] = out["sr_cena_m2"].pct_change() * 100
    out["skum_zmiana_proc"] = (out["sr_cena_m2"] / out["sr_cena_m2"].iloc[0] - 1) * 100
    return out


def missing_share(df: pd.DataFrame) -> pd.Series:
    return (df[["condition", "buildingMaterial", "buildYear"]].isna().mean() * 100).rename("braki_proc")


def correlations(df: pd.DataFrame, n: int = 10_000, n_seeds: int = 300) -> pd.DataFrame:
    """Pearson z ceną za m2. Próba 10 000 zależy od ziarna, więc liczę też pełny zbiór
    i rozrzut po 300 ziarnach, żeby było widać, ile ten wynik skacze."""
    cols = ["buildYear", "centreDistance"]
    full = df[cols + [TARGET]].corr()[TARGET][cols]
    one = df.sample(n, random_state=SEED)[cols + [TARGET]].corr()[TARGET][cols]
    many = np.array([
        df.sample(n, random_state=s)[cols + [TARGET]].corr()[TARGET][cols].to_numpy()
        for s in range(n_seeds)
    ])
    return pd.DataFrame({
        "pelny_zbior": full, f"proba_10000_seed{SEED}": one,
        "min_po_ziarnach": many.min(axis=0), "max_po_ziarnach": many.max(axis=0),
    })


def main(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = prepare(raw_dir)
    results = {
        "tab_X1": table_x1(df), "tab_X2": table_x2(df),
        "braki": missing_share(df).to_frame(), "korelacje": correlations(df),
    }
    for name, table in results.items():
        table.to_csv(os.path.join(out_dir, f"{name}.csv"))
        print(f"\n== {name} ==\n{table.round(3).to_string()}")
    print(f"\nN = {len(df)}, unikalnych ofert (id) = {df['id'].nunique()}, miast = {df['city'].nunique()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
