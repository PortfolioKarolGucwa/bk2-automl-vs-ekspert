# Prompt do wersji AI

Model: Claude Haiku. Osobna sesja, bez danych i bez żadnej informacji o rozdziale
ani o BK2%. Model nie uruchamiał kodu.

Były dwa podejścia. W pierwszym nie wskazałem algorytmu i wyszedł gradient boosting
(`src/ai_gradient_boosting.py`). W drugim poprosiłem wprost o Random Forest, bo jest
w tab. X.3 (`src/ai_random_forest.py`). Pliki mają tu inne nazwy niż w sesji
z modelem, stąd `model_haiku.py` w komunikacie o użyciu w pierwszym skrypcie.

```
Masz 11 plików CSV z ofertami sprzedaży mieszkań w 15 polskich miastach:
apartments_pl_2023_08.csv ... apartments_pl_2024_06.csv (jeden plik na miesiąc).

Kolumny: id, city, type, squareMeters, rooms, floor, floorCount, buildYear,
latitude, longitude, centreDistance, poiCount, schoolDistance, clinicDistance,
postOfficeDistance, kindergartenDistance, restaurantDistance, collegeDistance,
pharmacyDistance, ownership, buildingMaterial, condition, hasParkingSpace,
hasBalcony, hasElevator, hasSecurity, hasStorageRoom, price.

Napisz jeden skrypt w Pythonie (pandas, numpy, scikit-learn), który:
1. wczytuje i łączy pliki z katalogu podanego jako pierwszy argument wiersza poleceń,
2. przygotowuje dane do modelowania,
3. buduje model przewidujący cenę za metr kwadratowy,
4. ocenia go 5-krotną walidacją krzyżową i wypisuje R², RMSE i MAE,
5. zapisuje predykcje z walidacji do pliku predictions.csv w katalogu podanym jako drugi
   argument (kolumny: id, month, actual, predicted; month w formacie RRRR-MM).

Sam zdecyduj o czyszczeniu danych, doborze cech i modelu.
```

W drugim podejściu punkt 3 brzmiał „buduje model Random Forest przewidujący cenę za
metr kwadratowy”. Doszedł też wymóg układu kodu (funkcje `load_all`, `clean`,
`build_model`, stałe `FEATURES` i `TARGET`), żeby dało się go wołać z `evaluate.py`.
