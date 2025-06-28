import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(10)  

def create_dataset(ds, look_back=1):
    X_data, y_data = [], []
    for i in range(len(ds) - look_back):
        X_data.append(ds[i:(i + look_back), 0])
        y_data.append(ds[i + look_back, 0])
    return np.array(X_data), np.array(y_data)

look_back = 36  

df_test = pd.read_csv('C:/Users/User/Desktop/RNNLSTM/BTC LSTM/data/BTC_USDT_Test_2020-01-01_to_2020-04-30_4h.csv')

df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test.set_index('Date', inplace=True)
X_test_set = df_test[['Close']].values

sc = MinMaxScaler()
X_test_set = sc.fit_transform(X_test_set)

model = load_model('C:/Users/user/Desktop/RNNLSTM/BTC LSTM/data/Train_data/bitcoin_LSTM_model' + str(look_back) + '.keras')

predicted_prices = []
start_index = len(X_test_set) - len(df_test['2020-04-01':])

while start_index + look_back < len(X_test_set):
    X_test = X_test_set[start_index:start_index + look_back].reshape(1, look_back, 1)
    predicted_price = model.predict(X_test)

    predicted_prices.append(predicted_price[0, 0])
    
    start_index += 1

predicted_prices = np.array(predicted_prices).reshape(-1, 1)
predicted_prices = sc.inverse_transform(predicted_prices)


actual_prices = df_test['Close'].values[-len(predicted_prices):]

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

df_test['5 Day MA'] = df_test['Close'].rolling(window=5).mean()
df_test['10 Day MA'] = df_test['Close'].rolling(window=10).mean()

ma_5 = df_test['5 Day MA'].values[-len(predicted_prices):]
ma_10 = df_test['10 Day MA'].values[-len(predicted_prices):]

plt.figure(figsize=(12, 6))

plt.plot(df_test.index[-len(predicted_prices):], actual_prices, color="orange", label="Real Bitcoin Price (April)")
plt.plot(df_test.index[-len(predicted_prices):], predicted_prices, color="green", label="Predicted Bitcoin Price (April)")

plt.plot(df_test.index[-len(predicted_prices):], ma_5, color="blue", linestyle='--', label="5 Day Moving Average")
plt.plot(df_test.index[-len(predicted_prices):], ma_10, color="red", linestyle='--', label="10 Day Moving Average")

plt.title("2020 April Bitcoin Price: Real vs Predicted (LSTM) with Moving Averages")
plt.xlabel("Time")
plt.ylabel("Bitcoin Price")
plt.legend()
plt.show()
