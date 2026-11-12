"""Zestawienie: liczby z rozdziału (config/paper_targets.json) obok tego, co wychodzi z kodu.

Uruchomienie (po descriptive.py i evaluate.py):
    python src/compare_with_paper.py config/paper_targets.json results
"""
import json
import os
import sys

import pandas as pd

TOL_MODEL = 0.01   # metryki modeli uznaję za zgodne przy różnicy do 1%


def fmt(x: float) -> str:
    return f"{x:,.3f}".replace(",", " ") if abs(x) < 10 else f"{x:,.1f}".replace(",", " ")


def row(label, paper, code, exact_digits=None, tol=None):
    if exact_digits is not None:
        ok = round(code, exact_digits) == round(paper, exact_digits)
    else:
        ok = abs(code - paper) <= tol * abs(paper)
    return f"| {label} | {fmt(paper)} | {fmt(code)} | {'zgodne' if ok else 'ROZBIEŻNOŚĆ'} |", ok


def section(title, rows):
    lines = [f"\n## {title}\n", "| Pozycja | Rozdział | Kod | Ocena |", "|---|---|---|---|"]
    lines += [r for r, _ in rows]
    n_ok = sum(ok for _, ok in rows)
    lines.append(f"\nZgodnych: {n_ok} z {len(rows)}.")
    return lines


def main(targets_path: str, res: str) -> None:
    with open(targets_path, encoding="utf-8") as fh:
        t = json.load(fh)
    read = lambda name, **kw: pd.read_csv(os.path.join(res, name), **kw)
    out = ["# Rozdział a dane: zestawienie liczb"]

    x1 = read("tab_X1.csv", index_col=0)
    digits = {"liczba_obs": 0, "sr_cena_m2": 0, "min_cena": 0, "max_cena": 0, "sr_pow_m2": 1, "sr_odl_centrum_km": 2}
    out += section("Tab. X.1", [row(f"{c}: {k}", v, x1.loc[c, k], exact_digits=digits[k])
                                for c, d in t["tab_X1"].items() for k, v in d.items()])

    x2 = read("tab_X2.csv", index_col=0)
    digits = {"sr_cena_m2": 0, "liczba_obs": 0, "zmiana_mm_proc": 1, "skum_zmiana_proc": 1}
    out += section("Tab. X.2", [row(f"{p}: {k}", v, x2.loc[p, k], exact_digits=digits[k])
                                for p, d in t["tab_X2"].items() for k, v in d.items()])

    miss = read("braki.csv", index_col=0)["braki_proc"]
    out += section("Braki danych (%)", [row(k, v, miss[k], exact_digits=1) for k, v in t["braki"].items()])

    corr = read("korelacje.csv", index_col=0)
    lines = ["\n## Korelacje Pearsona z ceną za m² (próba 10 000)\n",
             "| Zmienna | Rozdział | Zakres po 300 ziarnach | Pełny zbiór | Ocena |", "|---|---|---|---|---|"]
    for k, v in t["korelacje"].items():
        lo, hi = corr.loc[k, "min_po_ziarnach"], corr.loc[k, "max_po_ziarnach"]
        verdict = "osiągalne" if lo <= v <= hi else "ROZBIEŻNOŚĆ (poza zakresem)"
        lines.append(f"| {k} | {v:.3f} | {lo:.3f} – {hi:.3f} | {corr.loc[k, 'pelny_zbior']:.3f} | {verdict} |")
    out += lines

    for scheme, desc in [("kfold", "losowy podział 5-krotny, jak w rozdziale"),
                         ("group", "podział 5-krotny po id oferty")]:
        x3 = read(f"tab_X3_{scheme}.csv").set_index("model")
        out += section(f"Tab. X.3 ({desc}; tolerancja {TOL_MODEL:.0%})",
                       [row(f"{m}: {k}", v, x3.loc[m, k], tol=TOL_MODEL)
                        for m, d in t["tab_X3"].items() for k, v in d.items()])
        out.append("\nWszystkie warianty z tego przebiegu:\n")
        out.append(x3.round({"R2": 3, "RMSE": 0, "MAE": 0, "czas_obliczen_s": 0}).to_markdown())
        out.append("\nN jest mniejsze niż 195 568, bo skrypty AI same obcięły skrajne ceny za m².")

        x4 = read(f"tab_X4_{scheme}.csv").set_index(["model", "podokres"])["RMSE"]
        rows = []
        for p, d in t["tab_X4"].items():
            rows.append(row(f"{p}: AutoML", d["automl"], x4["Gradient Boosting - AutoML", p], tol=TOL_MODEL))
            rows.append(row(f"{p}: hybryda", d["hybryda"], x4["GB + zmienne BK2% - hybrydowy", p], tol=TOL_MODEL))
        out += section(f"Tab. X.4 ({desc})", rows)
        out.append("\nRMSE w podokresach, wszystkie warianty:\n")
        out.append(x4.unstack("podokres").round(0).to_markdown())
        bias = read(f"tab_X4_{scheme}.csv").set_index(["model", "podokres"])["sredni_blad_proc"]
        out.append("\nŚredni błąd w podokresach (%; plus = model zawyża):\n")
        out.append(bias.unstack("podokres").round(1).to_markdown())

    fwd = read("test_czasowy.csv")
    out.append("\n## Test z podziałem czasowym (uczenie na miesiącach wcześniejszych)\n")
    out.append("RMSE w PLN/m²; średni błąd w %, minus = model zaniża.\n")
    out.append(fwd.pivot(index="miesiac", columns="model", values=["RMSE", "sredni_blad_proc"]).round(1).to_markdown())

    path = os.path.join(res, "raport_zgodnosci.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
