from datetime import date, timedelta
import numpy as np

import pandas as pd
import tensorflow as tf
import yfinance as yf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

tensorboard = TensorBoard(
    log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True
)
indices_asia_and_wig = [
    "^GSPC",
    "^N225",
    "^HSI",
    "^AXJO",
    "^KS11",
    "^BSESN",
    "^STI",
    "^TWII",
    "^KLSE",
    "^JKSE",
    "^NZ50",
    "^NSEI",
    "^BSESN",
    "^TWII",
    "^TWII",
]

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=100,
    verbose=4,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


def main():
    # download_yf_data(
    #     indices_asia_and_wig,
    #     "2000-01-01",
    #     0.95,
    #     0.95,
    # )
    df = load_data("data_asia_session.csv")
    df = clean_data(df)
    df_train = df[:-50]  # Use all rows except the last 30 rows for training
    df_predict = df[-50:]  # Use the last 30 rows for prediction
    print(df_predict)  # Print
    print(df_train)
    train_model(df_train, df_predict)


def remove_rows_and_columns_with_missing_values(
    df, row_threshold=0.95, col_threshold=0.95
):
    # Remove rows with missing values
    row_threshold_value = df.shape[1] * row_threshold
    df = df.dropna(thresh=row_threshold_value, axis=0)

    # Calculate the percentage of missing values in each column
    missing_percent = df.isnull().mean()

    # Select columns where the percentage of missing values is less than or equal to col_threshold
    selected_columns = missing_percent[missing_percent <= col_threshold].index

    # Remove unselected columns
    df = df[selected_columns]

    return df


def download_yf_data(list_index, start_date, rows_pct_notna, columns_pct_notna):
    stock_data = yf.download(
        list_index, interval="1d", start=start_date, ignore_tz=True
    )[["Close"]]
    print(stock_data)

    # Drop First Level of Column Label ( unnecessary 'Close' label )
    stock_data.columns = stock_data.columns.droplevel(0)

    # Remove rows and columns with missing values (here must be 95% of values)
    stock_data = remove_rows_and_columns_with_missing_values(
        stock_data, rows_pct_notna, columns_pct_notna
    )
    stock_data.dropna(inplace=True)
    # print data
    print(stock_data)

    # Save to CSV
    stock_data.to_csv("data_asia_session.csv")


def load_data(file):
    df = pd.read_csv(file, index_col=0)
    return df


def clean_data(df):
    print(df)
    df = df.pct_change()
    df.dropna(inplace=True)
    df["^GSPC"] = np.where(df["^GSPC"] > 0, 1, 0)
    return df


def train_model(df, df2):
    scaler = MinMaxScaler()

    y = df["^GSPC"]
    X = scaler.fit_transform(df.drop(["^GSPC"], axis=1))

    # second set for prediction
    y_2_to_PREDICT = df2["^GSPC"]
    X_df2_to_PREDICT = scaler.transform(df2.drop(["^GSPC"], axis=1))

    # Podział danych na zestawy treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    # Tworzenie modelu sieci neuronowej
    model = Sequential()
    model.add(Dense(1024, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dropout(0.001))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.001))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.001))

    model.add(Dense(1, activation="sigmoid"))

    learning_rate = 0.001

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Trenowanie modelu
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=320,
        validation_data=(X_test, y_test),
        callbacks=callback,
    )
    # Wykonanie prognozy na podstawie danych testowych

    # Pobranie 30 ostatnich rekordów z zestawu testowego
    X_last_30 = X_test[-30:]
    # Prognozowanie dla 30 ostatnich rekordów z zestawu testowego
    y_pred_last_30 = model.predict(X_last_30)
    y_pred_last_30 = [1 if y >= 0.5 else 0 for y in y_pred_last_30]

    # Pobranie odpowiednich wartości z zestawu testowego
    y_real_last_30 = y_test.tail(30).values

    # Tworzenie tabeli z prognozą i rzeczywistymi danymi
    print("Prediction for last 30 days from testing set: ")
    result_table = pd.DataFrame({"Predicted": y_pred_last_30, "Real": y_real_last_30})
    result_table.columns = ["Predicted", "Real"]
    # Dodanie date index
    stock_data = df.tail(30)
    result_table["Date"] = stock_data.index

    # tworzenie kolumny z dokładnością
    result_table["Accuracy"] = (
        result_table["Predicted"] == result_table["Real"]
    ).astype(int)

    # Wyświetlenie tabeli
    print(result_table)
    print(
        "Acurency last 30 days ",
        result_table["Accuracy"].sum() / len(result_table["Accuracy"]),
    )

    pred_value = model.predict(X_df2_to_PREDICT)

    prediction_y_pred = [1 if y >= 0.5 else 0 for y in pred_value]

    accuracy = (prediction_y_pred == y_2_to_PREDICT.values).astype(int)

    real_prediction_df = pd.DataFrame(
        {
            "prediction": prediction_y_pred,
            "real": y_2_to_PREDICT.values,
            "accuracy": accuracy,
        },
        index=df2.index,
    )
    print(real_prediction_df)
    print(
        "Acurency last 50 days ",
        real_prediction_df["accuracy"].sum() / len(real_prediction_df["accuracy"]),
    )


def load_last_data(list_index):
    list_index.remove("^GSPC")
    today = date.today()
    # biore 7 dni aby móc przewidzieć cenę
    daysback = today - timedelta(days=7)
    stock_data = yf.download(list_index, start=daysback, ignore_tz=True)[["Close"]]
    stock_data.columns = stock_data.columns.droplevel(0)
    null_counts = (
        stock_data.isna().sum() / stock_data.count()[0]
    )  # print fraction of NAN rows
    print("nulls:", null_counts)
    stock_data.fillna(method="ffill", inplace=True)
    df = stock_data.pct_change()
    print(df)
    print(df.tail(1))
    return df.tail(1)


if __name__ == "__main__":
    main()

    # # start_date='2000-01-01'
    # df = load_data(tickers_azja)
    # df2 = load_last_data(tickers_azja)
    # pred_this(df, df2)
