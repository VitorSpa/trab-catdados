import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_sequences(df_x, df_y, sequence_length):
    x, y = [], []
    for i in range(len(df_x) - sequence_length):
        x = df_x[i:(i + sequence_length)]
        y.append(df_y[i + sequence_length])
        x.append(x)
    return np.array(x), np.array(y)


def build_data(df):
    scaler = MinMaxScaler((-1, 1))
    train_cut = int(len(df) * 0.80)
    val_cut = int(len(df) * 0.90)

    df_train = df.iloc[:train_cut].copy()
    df_val = df.iloc[train_cut:val_cut].copy()
    df_test = df.iloc[val_cut:].copy()

    df_train['sRTT'] = scaler.fit_transform(df_train[['RTT']])
    df_val['sRTT'] = scaler.transform(df_val[['RTT']])
    df_test['sRTT'] = scaler.transform(df_test[['RTT']])

    sequence_length = 20
    X_train, y_train = create_sequences(df_train['sRTT'].values, df_train['sRTT'].values, sequence_length)
    X_val, y_val = create_sequences(df_val['sRTT'].values, df_val['sRTT'].values, sequence_length)
    X_test, y_test = create_sequences(df_test['sRTT'].values, df_test['sRTT'].values, sequence_length)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_val, y_val, X_test, y_test


UNITS_VALUE = 64


def gru_model(shape_1, shape_2):
    model = keras.Sequential([
        keras.layers.GRU(units=UNITS_VALUE, return_sequences=True, input_shape=(shape_1, shape_2)),
        keras.layers.GRU(units=UNITS_VALUE, return_sequences=True),
        keras.layers.GRU(units=UNITS_VALUE, return_sequences=True),
        keras.layers.GRU(units=UNITS_VALUE, return_sequences=True),
        keras.layers.GRU(units=UNITS_VALUE, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def rnn_model(shape_1, shape_2):
    model = keras.Sequential([
        keras.layers.SimpleRNN(units=UNITS_VALUE, return_sequences=True, input_shape=(shape_1, shape_2)),
        keras.layers.SimpleRNN(units=UNITS_VALUE, return_sequences=True),
        keras.layers.SimpleRNN(units=UNITS_VALUE, return_sequences=True),
        keras.layers.SimpleRNN(units=UNITS_VALUE, return_sequences=True),
        keras.layers.SimpleRNN(units=UNITS_VALUE, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def lstm_model(shape_1, shape_2):
    model = keras.Sequential([
        keras.layers.LSTM(units=UNITS_VALUE, return_sequences=True, input_shape=(shape_1, shape_2)),
        keras.layers.LSTM(units=UNITS_VALUE, return_sequences=True),
        keras.layers.LSTM(units=UNITS_VALUE, return_sequences=True),
        keras.layers.LSTM(units=UNITS_VALUE, return_sequences=True),
        keras.layers.LSTM(units=UNITS_VALUE, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    BATCH_SIZE = 32
    EPOCHS = 200

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=30,
                                                      mode='min')
    model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[early_stopping],
              validation_data=(X_val, y_val),
              verbose=0)
    return model


def model_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_test_flatten = y_test.flatten()
    y_pred_flatten = y_pred.flatten()
    mse = mean_squared_error(y_test_flatten, y_pred_flatten)
    rmse = mean_squared_error(y_test_flatten, y_pred_flatten, squared=False)
    mae = mean_absolute_error(y_test_flatten, y_pred_flatten)
    r2 = r2_score(y_test_flatten, y_pred_flatten)
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'R2 Score: {r2}')


def run_models(X_train, y_train, X_val, y_val, X_test, y_test):
    model1 = gru_model(X_train.shape[1], X_train.shape[2])
    model1 = train_model(model1, X_train, y_train, X_val, y_val)

    model2 = rnn_model(X_train.shape[1], X_train.shape[2])
    model2 = train_model(model2, X_train, y_train, X_val, y_val)

    model3 = lstm_model(X_train.shape[1], X_train.shape[2])
    model3 = train_model(model3, X_train, y_train, X_val, y_val)

    print('Eval. Model GRU:')
    model_eval(model1, X_test, y_test)
    print()
    print('Eval. Model RNN:')
    model_eval(model2, X_test, y_test)
    print()
    print('Eval. Model LSTM:')
    model_eval(model3, X_test, y_test)


# Read data
df = pd.read_csv('data__rtt.csv')

# Run model
X_train, y_train, X_val, y_val, X_test, y_test = build_data(df)
run_models(X_train, y_train, X_val, y_val, X_test, y_test)
