import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SimpleRNN
import matplotlib.pyplot as plt

df_train = pd.read_csv('C:/Users/User/Desktop/RNNLSTM/BTC RNN/data/BTC_USDT_Train_2018-01-01_to_2019-12-31_4h.csv',
                       index_col='Date', parse_dates=True)

def create_dataset(ds, look_back=36):  
    X_data, y_data = [], []
    for i in range(len(ds) - look_back):
        X_data.append(ds[i:(i + look_back), 0])
        y_data.append(ds[i + look_back, 0])
    return np.array(X_data), np.array(y_data)

X_train_set = df_train[['Close']].values


look_back = 36  

X_train, y_train = create_dataset(X_train_set, look_back)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

y_train_scaler = MinMaxScaler()
y_train = y_train_scaler.fit_transform(y_train.reshape(-1, 1))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(10, return_sequences=True, activation="tanh"))
model.add(Dropout(0.2))
model.add(SimpleRNN(5, return_sequences=False, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

model.fit(X_train, y_train, epochs=30, batch_size=32)

print("Saving Model: bitcoin_model" + str(look_back) + ".keras ...")
model.save('C:/Users/User/Desktop/RNNLSTM/BTC RNN/data/Train_data/bitcoin_RNN_model' + str(look_back) + '.keras')
