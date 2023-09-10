import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Pobranie danych historycznych SP500
data = yf.download("^IXIC", start="2000-01-01")

# Tworzenie kolumny z procentowymi zmianami cen
data["PriceChange"] = data["Close"].pct_change()

# Usunięcie wierszy z brakującymi danymi
data = data.dropna()

# Tworzenie zmiennych zero-jedynowych na podstawie kierunku ruchu cen
data["Label"] = (data["PriceChange"] > 0).astype(int)

# Zamiana cech miesiąca, dnia tygodnia i dnia miesiąca na kodowanie "one-hot"
data["Month"] = data.index.month
data["DayOfWeek"] = data.index.dayofweek
data["DayOfMonth"] = data.index.day
data = pd.get_dummies(data, columns=["Month", "DayOfWeek", "DayOfMonth"], dtype=int)


# Usunięcie kolumny z danymi procentowymi zmian cen
data = data[
    ["Label"]
    + [
        col
        for col in data.columns
        if col.startswith("Month_")
        or col.startswith("DayOfWeek_")
        or col.startswith("DayOfMonth_")
    ]
]

print(data)
# Podział danych na zestawy treningowe i testowe
X = data.iloc[:-1, 1:]  # Wejście modelu (cechy)
y = (
    data["Label"].shift(-1).dropna()
)  # Wyjście modelu (0 lub 1 w zależności od kierunku ruchu cen następnego dnia)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Utworzenie modelu Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Definicja hiperparametrów do optymalizacji
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Użycie GridSearchCV do znalezienia najlepszych hiperparametrów
grid_search = GridSearchCV(
    estimator=rf_model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Wybranie najlepszego modelu
best_rf_model = grid_search.best_estimator_

# Trenowanie najlepszego modelu
best_rf_model.fit(X_train, y_train)

# Przewidywanie
y_pred = best_rf_model.predict(X_test)

# Ocena modelu
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")
