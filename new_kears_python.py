from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


# Wczytanie danych z Yahoo Finance
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# Tworzenie sekwencji danych
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : i + sequence_length]
        target = data[i + sequence_length : i + sequence_length + 1]
        sequences.append((sequence, target))
    return sequences


# Podział danych na zbiór treningowy i testowy
def split_data(sequences, split_ratio):
    split_index = int(len(sequences) * split_ratio)
    train_data = sequences[:split_index]
    test_data = sequences[split_index:]
    return train_data, test_data


# Tworzenie modelu sieci neuronowej
def create_model(sequence_length):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(50, input_shape=(sequence_length, 1)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Trening modelu
def train_model(model, train_data, batch_size, epochs):
    X_train, y_train = zip(*train_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callback,
        verbose=2,
    )


# Ocena modelu na danych testowych
def evaluate_model(model, test_data, scaler):
    X_test, y_test = zip(*test_data)
    X_test = np.array(X_test)

    # Przygotuj dane testowe w odpowiedniej formie
    y_test = np.array(y_test)[:, -1]  # Wybierz tylko ostatnią wartość z sekwencji
    y_pred = model.predict(X_test)

    # Odwróć transformację danych
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_original, y_pred_original)
    return mse


today = date.today()
# Parametry
ticker = "^GSPC"
start_date = "2020-01-01"
end_date = today
sequence_length = 30
split_ratio = 0.8
batch_size = 32
epochs = 1000

# Wczytanie danych
data = load_data(ticker, start_date, end_date)
data = data[["Adj Close"]]  # Wybierz tylko kolumnę 'Adj Close'
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequences = create_sequences(scaled_data, sequence_length)
train_data, test_data = split_data(sequences, split_ratio)

# Tworzenie i trenowanie modelu
model = create_model(sequence_length)
train_model(model, train_data, batch_size, epochs)

# Ocena dokładności modelu na danych testowych
mse = evaluate_model(model, test_data, scaler)
print(f"Średni błąd kwadratowy (MSE) na danych testowych: {mse}")

# Wykres predykcji
X_test, y_test = zip(*test_data)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_pred = model.predict(X_test)

# Odwróć transformację danych dla wykresu
y_test_original = scaler.inverse_transform(y_test[:, -1].reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

plt.figure(figsize=(12, 6))
plt.plot(
    np.arange(len(y_test_original)),
    y_test_original,
    label="Rzeczywiste ceny",
    color="blue",
)
plt.plot(
    np.arange(len(y_test_original)),
    y_pred_original,
    label="Przewidywane ceny",
    color="red",
)
plt.xlabel("Dni")
plt.ylabel("Cena")
plt.legend()
plt.show()
