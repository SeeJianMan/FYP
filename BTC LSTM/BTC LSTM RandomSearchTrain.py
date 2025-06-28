import numpy as np
import pandas as pd
import joblib
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Dataset creation
def create_dataset(ds, look_back=1):
    X_data, y_data = [], []
    for i in range(len(ds) - look_back):
        X_data.append(ds[i:(i + look_back), 0])
        y_data.append(ds[i + look_back, 0])
    return np.array(X_data), np.array(y_data)

# LSTM model builder
def build_model(input_shape, lstm1, lstm2, dropout):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(lstm1, return_sequences=True, activation="tanh"))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm2, return_sequences=False, activation="tanh"))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model

# === Initialization ===
np.random.seed(10)
look_back = 36

# Load training data
train_path = 'C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/BTC_USDT_Train_2018-01-01_to_2019-12-31_4h.csv'
df_train = pd.read_csv(train_path, index_col='Date', parse_dates=True)
X_train_raw = df_train[['Close']].values

# Normalize and sequence
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X_train_raw)
X_all, y_all = create_dataset(X_scaled, look_back)
X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))
y_all = y_all.reshape(-1, 1)

# Save scaler
joblib.dump(sc, 'C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/Train_data/minmax_scaler.pkl')

# Split data
split_index = int(len(X_all) * 0.8)
X_train, X_val = X_all[:split_index], X_all[split_index:]
y_train, y_val = y_all[:split_index], y_all[split_index:]

# === Random Search Parameter Options ===
lstm1_options = [64, 128, 256]
lstm2_options = [32, 64, 128]
dropout_options = [0.1, 0.2, 0.3]
batch_sizes = [16, 32]
epochs_options = [100, 150, 200]

# Generate 20 random combinations
param_grid = []
for _ in range(20):
    param_grid.append({
        'lstm1': random.choice(lstm1_options),
        'lstm2': random.choice(lstm2_options),
        'dropout': random.choice(dropout_options),
        'batch_size': random.choice(batch_sizes),
        'epochs': random.choice(epochs_options)
    })

best_rmse = float('inf')
best_model = None
best_config = None
results = []

# === Random Search Training Loop ===
for i, params in enumerate(param_grid):
    print(f"\n🔁 Training {i+1}/{len(param_grid)}: {params}")
    start_time = time.time()

    model = build_model((look_back, 1), params['lstm1'], params['lstm2'], params['dropout'])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=params['epochs'], batch_size=params['batch_size'],
              verbose=0, callbacks=[es])

    elapsed = time.time() - start_time
    y_pred = model.predict(X_val)
    y_pred_inv = sc.inverse_transform(y_pred)
    y_val_inv = sc.inverse_transform(y_val)
    rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))

    print(f"⏱ Time: {elapsed:.2f}s | RMSE: {rmse:.4f}")
    params['rmse'] = rmse
    results.append(params)

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_config = params.copy()

# Save the best model
model_path = f"C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/Train_data/bitcoin_LSTM_model{look_back}.keras"
best_model.save(model_path)

print("\n✅ Best configuration found:")
print(best_config)
print(f"✅ Best RMSE: {best_rmse:.4f}")
print(f"✅ Model saved to: {model_path}")
