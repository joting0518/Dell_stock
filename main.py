import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

df = pd.read_csv('/Users/chenruoting/Desktop/test/dell_stock/archive/DELL_stock_history.csv', sep='\t', index_col='Date', parse_dates=True)

scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Close']])

# scaled_data = df[['Open', 'High', 'Low', 'Dividends', 'Stock Splits', 'Close']]

#每個樣本包含過去60天的資料
time_steps = 60

sequences = []
labels = []
#  (60, 7)矩陣 時間序列數據轉換成矩陣形式是為了讓模型能夠有效地學習時間相依性
for i in range(len(scaled_data) - time_steps):
    seq = scaled_data[i:i + time_steps, 0:7]
    label = scaled_data[i + time_steps, 6]  # 使用 'Close' 列作為預測目標
    sequences.append(seq)
    labels.append(label)

sequences
labels
sequences = np.array(sequences)
labels = np.array(labels)
# .dtypes

model = Sequential() #1 -> time_steps筆，也就是用前time_steps天的資料預測time_steps＋1天的close值 2 -> 將所有特徵整合為矩陣(time_steps＊number of features)
model.add(LSTM(100, input_shape=(sequences.shape[1], sequences.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = Adam(learning_rate=1e-4)
model.compile(optimizer='adam', loss='mse')

model.fit(sequences, labels, epochs=10, batch_size=32)

predictions = model.predict(sequences)
print(predictions)
predictions = np.concatenate((predictions, predictions, predictions, predictions, predictions, predictions, predictions), axis=1)
predictions = scaler.inverse_transform(predictions)
print(predictions[:, 6])
#virtualenv venv
#source venv/bin/activate
#deactivate