import pickle
from sklearn.model_selection import train_test_split
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

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


# Create sequences data for training and prediction
def create_sequences(data, seq, forecast):
    sequences = []
    # sequence from 1 do data length - seq - forecast
    for i in range(len(data) - seq - forecast):
        # eg. for sequence 30 , and 7 days forecast
        # first sequence is data[1:30], second is data[31:60], and so
        sequence = data[i : i + seq]
        # first target is data[31:38], second is data[38:45],
        target = data[i + seq : i + seq + forecast]
        # sequences = [([1:30], [31:38]), ([31:60], [38:45]), ...]
        sequences.append((sequence, target))
    return sequences


def split_data(sequences, split_ratio):
    split_index = int(len(sequences) * split_ratio)
    train_data = sequences[:split_index]
    test_data = sequences[split_index:]
    return train_data, test_data


# Parameters
ticker = "^IXIC"
start_date = "2020-01-01"
split_ratio = 0.8
batch_size = 800
epochs = 100
backward_days = 7
forecast_days = 1


# Download the nasdaq index from Yahoo Finance and extract Close price
data = yf.download(ticker, start=start_date)["Close"]

# Change to percentage change an drop Na value
data["pct_change"] = data.pct_change()

# Drop rows with missing values
data = data.dropna()

# Normalize the data by scaling the values between 0 and 1
# scaler = MinMaxScaler(feature_range=(-1, 1))
# change pandas to numpy array by values method,
# and reshape to (rows, columns),
#  -1 is calculating automatically value to fit the data, 1 is for 1 column
# scaled_data = scaler.fit_transform(data["pct_change"].values.reshape(-1, 1))
# print(scaled_data)

# Create sequences of data for training and prediction
sequences = create_sequences(
    data["pct_change"].values.reshape(-1, 1), backward_days, forecast_days
)

# Splitting the sequences into training and testing sets
X = np.array([sequence[0] for sequence in sequences])
y = np.array([sequence[1] for sequence in sequences])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=False
)


# Define the LSTM model with gradient clipping
model = Sequential()

# Add the input layer with LSTM and gradient clipping
model.add(
    LSTM(
        512,
        input_shape=(backward_days, 1),
        return_sequences=True,
        kernel_regularizer=l2(0.0001),
    )
)

# Add additional LSTM layers with gradient clipping
model.add(Dense(256))
model.add(Dropout(0.02))


# Add the output layer with the Huber loss function
model.add(Dense(forecast_days))
# Define the optimizer with the specified learning rate and momentum
opt = Adam(clipnorm=1.0, learning_rate=0.0001, beta_1=0.5)
# Compile the model with the Huber loss
model.compile(optimizer=opt, loss="mean_squared_error")

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[callback],
)

# save the model
pickle.dump(model, open("model.pkl", "wb"))

# save the scaler
# pickle.dump(scaler, open("scaler.pkl", "wb"))


# Make predictions
predicted_values = model.predict(X_test)


# Reshape y_test to 2D
# y_test_2d = y_test.reshape(-1, forecast_days)

# Invert scaling for predicted values
# predicted_values = scaler.inverse_transform(predicted_values)
# Invert scaling for actual values
# y_test_2d = scaler.inverse_transform(y_test_2d)

# Plot the actual vs. predicted values for a specific test sample
sample_index = 0  # You can change this to visualize a different sample
plt.figure(figsize=(12, 6))
plt.plot(predicted_values[sample_index], label="Actual Values", marker="o")
plt.plot(predicted_values[sample_index], label="Predicted Values", marker="x")
plt.title("Actual vs. Predicted Values")
plt.xlabel("Day")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot Mean Squared Error (MSE) for each test sample
mse = np.mean(np.square(predicted_values - X_test), axis=1)
plt.figure(figsize=(12, 6))
plt.plot(mse, label="MSE", marker="o")
plt.title("Mean Squared Error (MSE) for Test Samples")
plt.xlabel("Sample Index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()
