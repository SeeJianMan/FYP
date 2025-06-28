import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

look_back = 36

# === Load Test Data ===
df_test = pd.read_csv('C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/BTC_USDT_Test_2020-01-01_to_2020-04-30_4h.csv')
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test.set_index('Date', inplace=True)
X_raw = df_test[['Close']].values

# === Load Scalers ===
sc_lstm = joblib.load('C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/Train_data/minmax_scaler.pkl')
sc_rnn  = joblib.load('C:/Users/User/Desktop/RNNLSTM/BTC RNN/data/Train_data/minmax_scaler.pkl')

X_lstm_scaled = sc_lstm.transform(X_raw)
X_rnn_scaled  = sc_rnn.transform(X_raw)

# === Sequence Creator ===
def create_seq(data, look_back):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back, 0])
    return np.array(X)

X_lstm = create_seq(X_lstm_scaled, look_back)
X_rnn  = create_seq(X_rnn_scaled, look_back)

# === Match Length ===
min_len = min(len(X_lstm), len(X_rnn))
X_lstm = X_lstm[:min_len]
X_rnn  = X_rnn[:min_len]

X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
X_rnn  = X_rnn.reshape((X_rnn.shape[0], X_rnn.shape[1], 1))

# === Load Models ===
model_lstm = load_model('C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/Train_data/bitcoin_LSTM_model36.keras')
model_rnn  = load_model('C:/Users/User/Desktop/RNNLSTM/BTC RNN/data/Train_data/bitcoin_RNN_model36.keras')

# === Predict ===
pred_lstm = model_lstm.predict(X_lstm)
pred_rnn  = model_rnn.predict(X_rnn)

# === Inverse Transform ===
pred_lstm = sc_lstm.inverse_transform(pred_lstm)
pred_rnn  = sc_rnn.inverse_transform(pred_rnn)

# === Residual Ensemble ===
residual = pred_rnn - pred_lstm
ensemble_pred = pred_lstm + 0.5 * residual  # adjust weight here

# === Actual Prices ===
actual_prices = X_raw[-len(ensemble_pred):]
date_index = df_test.index[-len(ensemble_pred):]

# === Filter for April 9 to May 1 ===
mask = (date_index >= pd.Timestamp("2020-04-09")) & (date_index <= pd.Timestamp("2020-05-01"))
filtered_dates = date_index[mask]
filtered_actual = actual_prices[mask]
filtered_pred = ensemble_pred[mask]

# === Evaluation ===
rmse = np.sqrt(mean_squared_error(filtered_actual, filtered_pred))
r2 = r2_score(filtered_actual, filtered_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(filtered_dates, filtered_actual, label='Actual', color='orange')
plt.plot(filtered_dates, filtered_pred, label='Ensemble (LSTM + RNN Residual)', color='green')
plt.title(f'Bitcoin Price Prediction (Residual Ensemble)\nRMSE: {rmse:.2f}, R²: {r2:.2f}')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
