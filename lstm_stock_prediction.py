import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

#Velg aksje
target_symbol = "FB"

train_data = pd.read_csv('train_data.csv', parse_dates=['Date'])
val_data = pd.read_csv('validation_data.csv', parse_dates=['Date'])
test_data = pd.read_csv('test_data.csv', parse_dates=['Date'])


train_data = train_data[train_data['Symbol'] == target_symbol].copy()
val_data = val_data[val_data['Symbol'] == target_symbol].copy()
test_data = test_data[test_data['Symbol'] == target_symbol].copy()


if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
    raise ValueError(f"Ingen data for aksjen {target_symbol} i disse settene.")


train_data.sort_values('Date', inplace=True)
val_data.sort_values('Date', inplace=True)
test_data.sort_values('Date', inplace=True)

train_data.set_index('Date', inplace=True)
val_data.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

#Beregne log-returns
def compute_log_return(df):
    df = df.copy()
    df['LogClose'] = np.log(df['Close'])
    df['Return'] = df['LogClose'].diff()  #Daglig logavkastning
    df.dropna(inplace=True)
    return df[['Return', 'Close']]  #Beholder Return og Close

train_data = compute_log_return(train_data)
val_data = compute_log_return(val_data)
test_data = compute_log_return(test_data)


X_train = train_data[['Return']].values
X_val = val_data[['Return']].values
X_test = test_data[['Return']].values

y_train = train_data['Return'].values
y_val = val_data['Return'].values
y_test = test_data['Return'].values

#Normalisering
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


y_scaler = MinMaxScaler(feature_range=(0,1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1,1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1,1))

window_size = 30

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

#LSTM-modell
input_layer = Input(shape=(window_size, 1))
x = LSTM(50, return_sequences=True)(input_layer)
x = Dropout(0.2)(x)
x = LSTM(50)(x)
x = Dropout(0.2)(x)
output_layer = Dense(1)(x)

model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50, batch_size=32,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=[early_stopping],
    verbose=1
)

test_loss = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Loss: {test_loss}")

pred_scaled = model.predict(X_test_seq)
pred = y_scaler.inverse_transform(pred_scaled).flatten()

#Lager sekvens av pris i test-data for å plotte opp mot predikert pris
test_close = test_data['Close'].values
test_dates = test_data.index.values


test_close = test_close[window_size:]
test_dates = test_dates[window_size:]
true_returns = y_test[window_size:]

#Rekonstruer pris fra returns
#price(t) = price(t-1)*exp(return(t))
reconstructed_price_true = [test_close[0]]
reconstructed_price_pred = [test_close[0]]

for i in range(1, len(true_returns)):
    current_price_true = reconstructed_price_true[-1]*np.exp(true_returns[i])
    current_price_pred = reconstructed_price_pred[-1]*np.exp(pred[i])
    reconstructed_price_true.append(current_price_true)
    reconstructed_price_pred.append(current_price_pred)

reconstructed_price_true = np.array(reconstructed_price_true)
reconstructed_price_pred = np.array(reconstructed_price_pred)

#Plot Return sammenlikning (frivillig)
plt.figure(figsize=(12,6))
plt.plot(test_dates, true_returns, label='True Return')
plt.plot(test_dates, pred, label='Predicted Return', linestyle='--')
plt.title(f'Return Prediction for {target_symbol}')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Plot Price sammenlikning
plt.figure(figsize=(12,6))
plt.plot(test_dates, reconstructed_price_true, label='True Price')
plt.plot(test_dates, reconstructed_price_pred, label='Predicted Price', linestyle='--')
plt.title(f'Stock Price Prediction for {target_symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
