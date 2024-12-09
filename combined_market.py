import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

#Aksjer av interesse
symbols_of_interest = ["AAPL","AMZN","BABA","BAC","C","CELG","D","DIS","FB","GE","GOOG",
                       "INTC","JPM","MCD","MSFT","PCLN","PG","T","WMT","XOM"]


train_data = pd.read_csv('train_data.csv', parse_dates=['Date'])
val_data = pd.read_csv('validation_data.csv', parse_dates=['Date'])
test_data = pd.read_csv('test_data.csv', parse_dates=['Date'])


train_data = train_data[train_data['Symbol'].isin(symbols_of_interest)].copy()
val_data = val_data[val_data['Symbol'].isin(symbols_of_interest)].copy()
test_data = test_data[test_data['Symbol'].isin(symbols_of_interest)].copy()


train_data.sort_values(by='Date', inplace=True)
val_data.sort_values(by='Date', inplace=True)
test_data.sort_values(by='Date', inplace=True)

train_data.set_index('Date', inplace=True)
val_data.set_index('Date', inplace=True)
test_data.set_index('Date', inplace=True)

#Beregne log-returns
def compute_log_return(df):
    df = df.copy()
    df['LogClose'] = np.log(df['Close'])
    df['Return'] = df['LogClose'].diff() #Daglig logavkastning
    df.dropna(inplace=True)
    return df #Beholder Return

train_data = compute_log_return(train_data)
val_data = compute_log_return(val_data)
test_data = compute_log_return(test_data)

def prepare_data(df, use_sentiment=True):
    #Forenkler feature selection basert på use_sentiment
    df = df.copy()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    if use_sentiment:
        features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Sentiment']].values
    else:
        features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].values

    labels = df['Return'].values
    symbols = df[['Symbol']].values
    return features, labels, symbols

def create_sequences(X, y, window_size=30):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

#Bygg LSTM-modell
def build_and_train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    input_dim = X_train_seq.shape[2]
    input_layer = Input(shape=(window_size, input_dim))
    x = LSTM(64, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32,
              validation_data=(X_val_seq, y_val_seq), callbacks=[early_stopping], verbose=1)
    return model

# Kjør modell med sentiment
use_sentiment = True
X_train_s, y_train_s, symbols_train_s = prepare_data(train_data, use_sentiment=use_sentiment)
X_val_s, y_val_s, symbols_val_s = prepare_data(val_data, use_sentiment=use_sentiment)
X_test_s, y_test_s, symbols_test_s = prepare_data(test_data, use_sentiment=use_sentiment)


encoder_s = OneHotEncoder(sparse=False, handle_unknown='ignore')
symbols_train_enc_s = encoder_s.fit_transform(symbols_train_s)
symbols_val_enc_s = encoder_s.transform(symbols_val_s)
symbols_test_enc_s = encoder_s.transform(symbols_test_s)

X_train_full_s = np.hstack([X_train_s, symbols_train_enc_s])
X_val_full_s = np.hstack([X_val_s, symbols_val_enc_s])
X_test_full_s = np.hstack([X_test_s, symbols_test_enc_s])

scaler_s = MinMaxScaler(feature_range=(0,1))
X_train_scaled_s = scaler_s.fit_transform(X_train_full_s)
X_val_scaled_s = scaler_s.transform(X_val_full_s)
X_test_scaled_s = scaler_s.transform(X_test_full_s)

y_scaler_s = MinMaxScaler(feature_range=(0,1))
y_train_scaled_s = y_scaler_s.fit_transform(y_train_s.reshape(-1,1))
y_val_scaled_s = y_scaler_s.transform(y_val_s.reshape(-1,1))
y_test_scaled_s = y_scaler_s.transform(y_test_s.reshape(-1,1))

window_size = 30
X_train_seq_s, y_train_seq_s = create_sequences(X_train_scaled_s, y_train_scaled_s, window_size)
X_val_seq_s, y_val_seq_s = create_sequences(X_val_scaled_s, y_val_scaled_s, window_size)
X_test_seq_s, y_test_seq_s = create_sequences(X_test_scaled_s, y_test_scaled_s, window_size)

model_s = build_and_train_lstm(X_train_seq_s, y_train_seq_s, X_val_seq_s, y_val_seq_s)
test_loss_s = model_s.evaluate(X_test_seq_s, y_test_seq_s, verbose=1)
print("Test Loss (with sentiment):", test_loss_s)

pred_scaled_s = model_s.predict(X_test_seq_s)
pred_s = y_scaler_s.inverse_transform(pred_scaled_s).flatten()

test_dates = test_data.index.values[window_size:]
true_returns = y_test_s[window_size:]
test_symbols = symbols_test_s[window_size:]
test_close = test_data['Close'].values[window_size:]

#Rekonstruer priser fra returns (faktisk)
reconstructed_price_true = [test_close[0]]
for i in range(1, len(true_returns)):
    reconstructed_price_true.append(reconstructed_price_true[-1]*np.exp(true_returns[i]))
reconstructed_price_true = np.array(reconstructed_price_true)

#Rekonstruer priser fra predikerte returns (med sentiment)
reconstructed_price_pred_s = [test_close[0]]
for i in range(1, len(pred_s)):
    reconstructed_price_pred_s.append(reconstructed_price_pred_s[-1]*np.exp(pred_s[i]))
reconstructed_price_pred_s = np.array(reconstructed_price_pred_s)

#Kjør modell uten sentiment
use_sentiment = False
X_train_ns, y_train_ns, symbols_train_ns = prepare_data(train_data, use_sentiment=use_sentiment)
X_val_ns, y_val_ns, symbols_val_ns = prepare_data(val_data, use_sentiment=use_sentiment)
X_test_ns, y_test_ns, symbols_test_ns = prepare_data(test_data, use_sentiment=use_sentiment)

encoder_ns = OneHotEncoder(sparse=False, handle_unknown='ignore')
symbols_train_enc_ns = encoder_ns.fit_transform(symbols_train_ns)
symbols_val_enc_ns = encoder_ns.transform(symbols_val_ns)
symbols_test_enc_ns = encoder_ns.transform(symbols_test_ns)

X_train_full_ns = np.hstack([X_train_ns, symbols_train_enc_ns])
X_val_full_ns = np.hstack([X_val_ns, symbols_val_enc_ns])
X_test_full_ns = np.hstack([X_test_ns, symbols_test_enc_ns])

scaler_ns = MinMaxScaler(feature_range=(0,1))
X_train_scaled_ns = scaler_ns.fit_transform(X_train_full_ns)
X_val_scaled_ns = scaler_ns.transform(X_val_full_ns)
X_test_scaled_ns = scaler_ns.transform(X_test_full_ns)

y_scaler_ns = MinMaxScaler(feature_range=(0,1))
y_train_scaled_ns = y_scaler_ns.fit_transform(y_train_ns.reshape(-1,1))
y_val_scaled_ns = y_scaler_ns.transform(y_val_ns.reshape(-1,1))
y_test_scaled_ns = y_scaler_ns.transform(y_test_ns.reshape(-1,1))

X_train_seq_ns, y_train_seq_ns = create_sequences(X_train_scaled_ns, y_train_scaled_ns, window_size)
X_val_seq_ns, y_val_seq_ns = create_sequences(X_val_scaled_ns, y_val_scaled_ns, window_size)
X_test_seq_ns, y_test_seq_ns = create_sequences(X_test_scaled_ns, y_test_scaled_ns, window_size)

model_ns = build_and_train_lstm(X_train_seq_ns, y_train_seq_ns, X_val_seq_ns, y_val_seq_ns)
test_loss_ns = model_ns.evaluate(X_test_seq_ns, y_test_seq_ns, verbose=1)
print("Test Loss (without sentiment):", test_loss_ns)

pred_scaled_ns = model_ns.predict(X_test_seq_ns)
pred_ns = y_scaler_ns.inverse_transform(pred_scaled_ns).flatten()

#Rekonstruer priser fra predikerte returns (uten sentiment)
reconstructed_price_pred_ns = [test_close[0]]
for i in range(1, len(pred_ns)):
    reconstructed_price_pred_ns.append(reconstructed_price_pred_ns[-1]*np.exp(pred_ns[i]))
reconstructed_price_pred_ns = np.array(reconstructed_price_pred_ns)

#Lager markedsindeks (likevektet gjennomsnitt) for hver dag
unique_test_days = np.unique(test_dates)
unique_test_symbols = test_data['Symbol'].unique()
N = len(unique_test_symbols)

#For å gjøre dette må vi dele opp reconstructed_price_* per symbol per dag
#All data er flatet ut i test_symbols / test_dates / ... 
#For hver dag kan vi finne masken for den dagen og ta gjennomsnittet over de aksjene den dagen

actual_index = []
pred_index_with_s = []
pred_index_without_s = []

for day in unique_test_days:
    day_mask = (test_dates == day)
    #Faktiske priser den dagen for alle aksjer
    daily_actual_prices = reconstructed_price_true[day_mask]
    daily_pred_prices_s = reconstructed_price_pred_s[day_mask]
    daily_pred_prices_ns = reconstructed_price_pred_ns[day_mask]

    #Beregner likevektet gjennomsnitt
    actual_index.append(np.mean(daily_actual_prices))
    pred_index_with_s.append(np.mean(daily_pred_prices_s))
    pred_index_without_s.append(np.mean(daily_pred_prices_ns))

actual_index = np.array(actual_index)
pred_index_with_s = np.array(pred_index_with_s)
pred_index_without_s = np.array(pred_index_without_s)

# Plotter indeksen
plt.figure(figsize=(12,6))
plt.plot(unique_test_days, actual_index, label='Actual Market Index', color='black')
plt.plot(unique_test_days, pred_index_with_s, label='Predicted Market Index (with sentiment)', linestyle='--', color='red')
plt.plot(unique_test_days, pred_index_without_s, label='Predicted Market Index (without sentiment)', linestyle='--', color='blue')
plt.title('Market Index Comparison')
plt.xlabel('Date')
plt.ylabel('Index Value')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()