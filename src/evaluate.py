"""Cztery modele z tab. X.3, RMSE w podokresach (tab. X.4) i test czasowy.

  Random Forest - AutoML        ai_random_forest.py, kod od AI bez zmian
  Gradient Boosting - AutoML    ai_gradient_boosting.py, kod od AI bez zmian
  Regresja liniowa - ekspercka  expert_pipeline.linear_regression
  GB + zmienne BK2% - hybrydowy GB od AI + bk2_active, panic_month

Dodatkowo dwa warianty kontrolne: GB z samym numerem miesiąca i GB z miesiącem
oraz zmiennymi BK2%. Chciałem sprawdzić, ile daje sama data.

Skrypty AI czyszczą dane po swojemu, więc każdy liczy się na próbie, którą sam
wybrał. Regresja i hybryda idą na próbie gradient boostingu, żeby dało się je
z nim porównać.

Uruchomienie:  python src/evaluate.py data/raw results
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

import ai_gradient_boosting as ai_gb
import ai_random_forest as ai_rf
import expert_pipeline as ex
from common import BK2, MONTH, TARGET, add_time_features, make_cv, metrics, rmse, subperiods

RF, GB, OLS, HYB = ("Random Forest - AutoML", "Gradient Boosting - AutoML",
                    "Regresja liniowa - ekspercka", "GB + zmienne BK2% - hybrydowy")
GB_MONTH, HYB_MONTH = "GB + miesiąc (wrażliwość)", "GB + miesiąc + zmienne BK2% (wrażliwość)"


def build_tasks(raw_dir: str) -> dict:
    """nazwa -> (funkcja budująca model, X, y, id, miesiąc)."""
    raw = ai_gb.load_all(Path(raw_dir))

    gb = add_time_features(ai_gb.clean(raw), "month")          # próba wybrana przez skrypt GB
    y_gb = gb["price_per_m2"].to_numpy()

    expert = raw.loc[gb.index].copy()                           # te same oferty, surowe kolumny
    assert (expert["id"].to_numpy() == gb["id"].to_numpy()).all()
    expert[MONTH] = gb[MONTH].to_numpy()

    rf = ai_rf.clean(raw)                                       # próba wybrana przez skrypt RF
    y_rf = rf[ai_rf.TARGET].to_numpy()

    def task(build, X, y, frame):
        return build, X, y, frame["id"].to_numpy(), frame["month"].reset_index(drop=True)

    return {
        GB: task(ai_gb.build_model, gb[ex.AI_GB_FEATURES], y_gb, gb),
        OLS: task(ex.linear_regression, expert[ex.OLS_FEATURES], y_gb, gb),
        HYB: task(ex.hybrid_gradient_boosting, gb[ex.HYBRID_FEATURES], y_gb, gb),
        GB_MONTH: task(lambda: ex.gb_with_extra([MONTH]), gb[ex.AI_GB_FEATURES + [MONTH]], y_gb, gb),
        HYB_MONTH: task(lambda: ex.gb_with_extra([MONTH] + BK2), gb[ex.AI_GB_FEATURES + [MONTH] + BK2], y_gb, gb),
        RF: task(ai_rf.build_model, rf[ai_rf.FEATURES], y_rf, rf),   # na końcu, bo mieli się ~10 min na jeden przebieg CV
    }


def cross_validate(tasks: dict, scheme: str, out_dir: str) -> None:
    overall, by_period = [], []
    for name, (build, X, y, ids, month) in tasks.items():
        start = time.perf_counter()
        pred = cross_val_predict(build(), X, y, cv=make_cv(scheme), groups=ids if scheme == "group" else None)
        seconds = time.perf_counter() - start
        overall.append({"model": name, "N": len(y), **metrics(y, pred), "czas_obliczen_s": seconds})
        for label, mask in subperiods(month).items():
            by_period.append({"model": name, "podokres": label, "RMSE": rmse(y[mask], pred[mask]),
                              "sredni_blad_proc": float(np.mean(pred[mask] - y[mask]) / y[mask].mean() * 100)})
        print(f"[{scheme}] {name}: N={len(y)} R2={overall[-1]['R2']:.3f} RMSE={overall[-1]['RMSE']:.0f} "
              f"MAE={overall[-1]['MAE']:.0f} ({seconds:.0f}s)", flush=True)
        pd.DataFrame(overall).to_csv(os.path.join(out_dir, f"tab_X3_{scheme}.csv"), index=False)
        pd.DataFrame(by_period).to_csv(os.path.join(out_dir, f"tab_X4_{scheme}.csv"), index=False)


def forward_test(tasks: dict, out_dir: str) -> None:
    """Uczenie na miesiącach przed t, prognoza miesiąca t.

    Tak wygląda to w praktyce: w grudniu 2023 nikt nie miał jeszcze ani jednej
    grudniowej oferty w zbiorze uczącym.
    """
    rows = []
    for name in [GB, HYB, GB_MONTH]:
        build, X, y, _, month = tasks[name]
        for period in sorted(month.unique())[1:]:
            train, test = (month < period).to_numpy(), (month == period).to_numpy()
            pred = build().fit(X[train], y[train]).predict(X[test])
            rows.append({"miesiac": period, "model": name, "RMSE": rmse(y[test], pred),
                         "sredni_blad_proc": float(np.mean(pred - y[test]) / y[test].mean() * 100)})
        print(f"[test czasowy] {name}: gotowe", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "test_czasowy.csv"), index=False)


def main(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tasks = build_tasks(raw_dir)
    forward_test(tasks, out_dir)
    for scheme in ["kfold", "group"]:
        cross_validate(tasks, scheme, out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
