# BK2% a wycena mieszkań: AutoML kontra ekspert

Kod do mojego rozdziału „Granice automatyzacji analitycznej w warunkach szoków
regulacyjnych. Studium przypadku polskiego rynku nieruchomości”. Oryginalne
notatniki przepadły, więc to jest odtworzenie eksperymentu od zera na tych samych
danych, a nie kod, z którego powstały tabele w rozdziale. Różnice opisuję niżej.

Dane to [Apartment Prices in Poland](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland)
z Kaggle (K. Jamroz): oferty sprzedaży z 15 miast, sierpień 2023 – czerwiec 2024,
195 568 wierszy.

## O co chodzi

Pytanie jest proste: czy model zbudowany „z automatu”, bez wiedzy o rynku, poradzi
sobie z okresem, w którym ceny podbił program Bezpieczny Kredyt 2%. Porównuję dwie
wersje.

**Wersja AI.** `src/ai_gradient_boosting.py` i `src/ai_random_forest.py` wygenerował
model językowy (Claude Haiku). Dostał listę plików, nazwy kolumn i polecenie
zbudowania modelu ceny za m². O BK2% nie wiedział nic. Kodu nie poprawiałem, bo cały
sens tej wersji polega na tym, że decyzje podejmuje maszyna. Prompt leży w
`PROMPT_AI.md`.

**Wersja ekspercka.** `src/expert_pipeline.py`. Regresja liniowa, w której każdą
decyzję o zmiennych da się uzasadnić, oraz model hybrydowy: gradient boosting
z wersji AI plus dwie zmienne, których w danych nie ma. `bk2_active` to okres
działania programu (VIII–XII 2023), `panic_month` to grudzień 2023, kiedy kończył
się limit.

## Jak to uruchomić

Pliki `apartments_pl_2023_08.csv` … `apartments_pl_2024_06.csv` trzeba wrzucić do
`data/raw/` (plików z najmem nie używam).

```
pip install -r requirements.txt
python src/descriptive.py data/raw results
python src/evaluate.py data/raw results
python src/compare_with_paper.py config/paper_targets.json results
```

Pierwszy skrypt liczy część opisową i trwa około minuty. `evaluate.py` to jakieś
35 minut na dwóch rdzeniach, z czego większość zjada Random Forest. Trzeci skrypt
składa `results/raport_zgodnosci.md`, czyli zestawienie moich wyników z liczbami
z rozdziału.

Skrypty AI można też puścić osobno:
`python src/ai_gradient_boosting.py data/raw results`.

## Co gdzie jest

- `src/common.py`: wczytanie plików, cena za m², zmienne BK2%, metryki, podział CV
- `src/descriptive.py`: tab. X.1 i X.2, braki danych, korelacje
- `src/evaluate.py`: 5-krotna CV, RMSE w podokresach, test z podziałem czasowym
- `src/compare_with_paper.py`: porównanie z rozdziałem
- `config/paper_targets.json`: liczby przepisane z rozdziału
- `results/`: wyniki ostatniego przebiegu

## Wyniki i zgodność z rozdziałem

Część opisowa wychodzi tak samo: N, cała tab. X.2 (w tym +6,8% w grudniu 2023),
odsetki braków, a w tab. X.1 liczebności, średnie ceny, powierzchnie i odległości
od centrum.

Nie zgadzają się ceny minimalne i maksymalne w tab. X.1, korelacja ceny
z odległością od centrum oraz metryki modeli. Losowa 5-krotna CV daje:

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest (AI) | 0,947 | 1 126 | 698 |
| Gradient boosting (AI) | 0,852 | 1 876 | 1 387 |
| Regresja liniowa (ekspercka) | 0,713 | 2 615 | 1 991 |
| GB + zmienne BK2% (hybryda) | 0,869 | 1 765 | 1 303 |

Hybryda jest lepsza od gradient boostingu AI o 111 PLN/m², w rozdziale było 164.
RMSE w podokresach (tab. X.4) nie odtwarza się w ogóle. Pełne zestawienie, liczba
po liczbie, jest w `results/raport_zgodnosci.md`.

## Na co trafiłem po drodze

Unikalnych ofert jest tylko 92 967. Reszta to te same mieszkania wiszące
w ogłoszeniach przez kolejne miesiące. Przy losowym podziale ta sama oferta ląduje
i w treningu, i w teście, na czym najbardziej korzysta Random Forest. Dlatego liczę
też podział po `id`; RF spada wtedy do R² = 0,883.

Oba skrypty AI same obcięły po 0,5% skrajnych cen z każdej strony, więc modele
uczą się na 193 625 obserwacjach.

Gradient boosting od AI w ogóle nie korzysta z daty oferty. Random Forest, pisany
w osobnym podejściu, numer miesiąca już sobie dodał. Ma to spore znaczenie:
zmienne BK2% pomagają modelowi, który nie zna czasu, ale sam numer miesiąca daje
praktycznie to samo (RMSE 1 749 wobec 1 765). W tych danych informacja o programie
i informacja o dacie są nie do rozdzielenia.

Najciekawszy wyszedł test czasowy, czyli uczenie na wcześniejszych miesiącach
i prognoza następnego. Gradient boosting AI po grudniowym skoku zaniża ceny
o około 6% i nie wraca do normy do końca okresu.

Czasów pracy z tab. X.5 kod nie mierzy, zapisuje tylko czas obliczeń.
