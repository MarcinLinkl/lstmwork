import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf

callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=40,
    verbose=4,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


# Ustaw seed dla powtarzalności wyników
np.random.seed(0)
tf.random.set_seed(0)


# Funkcja do obliczania procentowych zmian cen akcji
def calculate_percent_change(prices):
    return (prices / prices.shift(1) - 1).dropna()


# Pobieramy dane historyczne cen akcji S&P 500 i obliczamy procentowe zmiany
data = yf.download("^GSPC", start="2000-01-01")
percent_changes = calculate_percent_change(data["Close"])

# Tworzymy DataFrame z danymi historycznymi, aby ustawić odpowiedni indeks
data_df = pd.DataFrame(
    {"Percent Change": percent_changes.values[:-1]}, index=percent_changes.index[1:]
)

# Normalizujemy dane
scaler = MinMaxScaler()
percent_changes_scaled = scaler.fit_transform(percent_changes.values.reshape(-1, 1))

# Przygotowujemy dane do trenowania modelu
X = []
y = []
window_size = 7  # Okno 7 dni

for i in range(len(percent_changes_scaled) - window_size):
    X.append(percent_changes_scaled[i : i + window_size])
    y.append(percent_changes_scaled[i + window_size])

X = np.array(X)
y = np.array(y)

# Dzielimy dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tworzymy prosty model sieci neuronowej
model = keras.Sequential(
    [
        keras.layers.LSTM(50, activation="relu", input_shape=(window_size, 1)),
        keras.layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mean_squared_error")

# Trenujemy model
epochs = 10
model.fit(
    X_train, y_train, epochs=epochs, batch_size=32, verbose=1, callbacks=[callback]
)

# Ocena modelu na zbiorze testowym
loss = model.evaluate(X_test, y_test)
print(f"Loss on test data: {loss}")

# Przewidujemy procentowe zmiany cen akcji na kolejne 7 dni
last_7_days = percent_changes_scaled[-window_size:].reshape(1, window_size, 1)
predicted_percent_changes = []

how_many_days = 7
for _ in range(how_many_days):
    prediction = model.predict(last_7_days)[0][0]
    predicted_percent_changes.append(prediction)
    last_7_days = np.roll(last_7_days, -1)
    last_7_days[-1] = prediction

# Odwracamy normalizację
predicted_percent_changes = scaler.inverse_transform(
    np.array(predicted_percent_changes).reshape(-1, 1)
)

# Obliczamy przewidywane ceny akcji na kolejne dni na podstawie procentowych zmian
last_price = data["Close"].iloc[-1]
predicted_prices = [last_price]

for percent_change in predicted_percent_changes:
    next_price = predicted_prices[-1] * (1 + percent_change)
    predicted_prices.append(next_price)

# Tworzymy daty dla przewidywanych dni
last_date = data.index[-1]
predicted_dates = [
    last_date + pd.DateOffset(days=i) for i in range(1, how_many_days + 1)
]

# Wyświetlamy przewidywane ceny akcji na kolejne dni wraz z wykresem
plt.figure(figsize=(12, 6))
plt.plot(
    data_df.index,
    data_df["Percent Change"],
    label="Historical Percent Changes",
    color="b",
)
plt.plot(
    predicted_dates,
    predicted_prices[1:],  # Pomijamy pierwszą cenę, ponieważ jest ona znaną ceną
    marker="o",
    linestyle="-",
    color="r",
    label="Predicted Prices",
)
plt.title(f"Predicted Stock Prices for the Next {how_many_days} Days")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Wyświetlamy przewidywane procentowe zmiany cen akcji na kolejne dni
print(f"Przewidywane procentowe zmiany cen akcji na kolejne {how_many_days} dni:")
for i, (date, percent_change) in enumerate(
    zip(predicted_dates, predicted_percent_changes), start=1
):
    print(f"Dzień {i}: {date.date()} - Procentowa Zmiana: {percent_change[0]:.4f}")
