"""Wersja ekspercka.

1. Regresja liniowa (MNK), zmienne dobrane ręcznie.
2. Hybryda: gradient boosting z ai_gradient_boosting.py plus dwie zmienne o BK2%.
   Poza nimi nic w modelu nie ruszam.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import ai_gradient_boosting as ai_gb
from common import BK2, MONTH

# Dobór zmiennych do regresji:
# - condition wyrzucam, 74,8% braków to za dużo, żeby cokolwiek imputować
# - buildingMaterial ma 39,6% braków, brak idzie jako osobna kategoria
# - buildYear (16,5% braków) i floor: mediana w mieście + flaga, że był brak
# - latitude/longitude pomijam: liniowo to prawie to samo co miasto, a położenie
#   wewnątrz miasta i tak opisuje centreDistance i odległości do usług
# - monthIdx jako zwykły trend liniowy
OLS_NUMERIC = [
    "squareMeters", "rooms", "floor", "floorCount", "buildYear",
    "centreDistance", "poiCount", "schoolDistance", "clinicDistance",
    "postOfficeDistance", "kindergartenDistance", "restaurantDistance",
    "collegeDistance", "pharmacyDistance", MONTH,
]
OLS_CATEGORICAL = [
    "city", "type", "ownership", "buildingMaterial",
    "hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom",
]
OLS_FEATURES = OLS_NUMERIC + OLS_CATEGORICAL


class CityMedianImputer(BaseEstimator, TransformerMixin):
    """Braki uzupełniane medianą w mieście (liczoną tylko na części uczącej)
    plus flagi braku dla wybranych kolumn."""

    def __init__(self, columns, flag_columns=("buildYear", "floor")):
        self.columns = columns
        self.flag_columns = flag_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.city_median_ = X.groupby("city")[self.columns].median()
        self.global_median_ = X[self.columns].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.flag_columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
        by_city = self.city_median_.reindex(out["city"]).set_axis(out.index)
        out[self.columns] = out[self.columns].fillna(by_city).fillna(self.global_median_)
        return out


def linear_regression() -> Pipeline:
    numeric_to_impute = [c for c in OLS_NUMERIC if c != MONTH]
    flags = ["buildYear_missing", "floor_missing"]
    encode = ColumnTransformer([
        ("num", StandardScaler(), OLS_NUMERIC + flags),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="brak")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), OLS_CATEGORICAL),
    ])
    return Pipeline([
        ("impute", CityMedianImputer(numeric_to_impute)),
        ("encode", encode),
        ("model", LinearRegression()),
    ])


AI_GB_FEATURES = ai_gb.NUM_COLS + ai_gb.BOOL_COLS + ai_gb.CAT_COLS


def gb_with_extra(extra: list[str]) -> Pipeline:
    """Model z wersji AI z tymi samymi ustawieniami. Dopisuję tylko dodatkowe
    kolumny liczbowe do jego preprocesora."""
    model = ai_gb.build_model()
    name, transformer, columns = model.named_steps["prep"].transformers[0]
    model.named_steps["prep"].transformers[0] = (name, transformer, list(columns) + extra)
    return model


def hybrid_gradient_boosting() -> Pipeline:
    """GB z wersji AI + bk2_active i panic_month. Cała różnica względem wersji AI
    bierze się z tych dwóch zmiennych."""
    return gb_with_extra(BK2)


HYBRID_FEATURES = AI_GB_FEATURES + BK2
