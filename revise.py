# import必要套件
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# 讀入資料
file_path = '/Users/chenruoting/Desktop/test/dell_stock/archive/DELL_stock_history.csv'  
df = pd.read_csv(file_path, sep='\t', index_col='Date', parse_dates=True)

# 進行特徵縮放：如果有特徵跟其他特徵相比很大，容易影響模型訓練，會fit在一個值上。這裡是Volume特別大，故進行特徵縮放
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Close']])

# time_steps代表每個訓練樣本包含過去60天的股票數據，sequences則是訓練樣本，
# 一個sequence包含過去60天的股票數據，labels則是我們的目標：收盤價，延續前面的例子，第一個label就是第61天的收盤價
time_steps = 60
sequences = []
labels = []
# 轉換成 (60, 7)矩陣(時間序列數據轉換成矩陣形式是為了讓模型能夠有效地學習時間相依性)
for i in range(time_steps, len(scaled_data)):
    sequences.append(scaled_data[i-time_steps:i])
    labels.append(scaled_data[i, -1])  # 使用 'Close' 列作為預測目標

# list轉為numpy array方便餵給模型
sequences = np.array(sequences)
labels = np.array(labels)

# 這裡使用RNN架構的LSTM，他可以很好的學習時間序列相關的資料，因為他會將結果加回下次的預測當中（Recurrent Connections）
# 參考資料：https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_rnns_lstm_work.html
# 開始建立模型
# model總共有 7 層：3 層 LSTM，3 層 Dropout，和 1 層 Dense
# LSTM有100個神經元，sequences.shape[1], sequences.shape[2]分別是 (60, 7)，代表time_steps = 60和sequences的feature ＝ 7
# dropout是用來防止overfitting。由於神經元不能依賴於其他神經元的存在，它們被迫學習更加穩健的特徵，這有助於減少模型對訓練數據的overfitting
# dense是一個全連接層，有一個神經元，用來輸出預測結果
model = Sequential()
model.add(LSTM(100, input_shape=(sequences.shape[1], sequences.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))

# 將optimizer加入去幫助模型自度調整參數，loss function加入去衡量模型效果
opt = Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='mse')
# 訓練開始 epochs=10代表會go through 資料集10次，batch_size=32代表每次傳遞給模型進行訓練的資料個數
model.fit(sequences, labels, epochs=10, batch_size=32)
# 使用訓練好的模型對訓練樣本進行預測
predictions = model.predict(sequences)
# 將預測結果與原始數據的其他特徵合併，然後逆縮放操作便可以獲得原始的股票價格
predictions = scaler.inverse_transform(np.hstack((scaled_data[time_steps:, :-1], predictions))) 
# 印出模型預測的收盤價
print(predictions[:, -1])

import matplotlib.pyplot as plt
# 要從第60筆資料開始比較因為time_steps = 60

# 逆縮放實際的收盤價
actual_prices = scaler.inverse_transform(np.hstack((scaled_data[time_steps:, :-1], labels.reshape(-1, 1))))[:, -1]

# 確保預測和實際價格數據的長度相同
min_length = min(len(predictions[:, -1]), len(actual_prices))
predicted_prices = predictions[:min_length, -1]
actual_prices = actual_prices[:min_length]

# 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(predicted_prices, label='predicted_prices', color='blue')
plt.plot(actual_prices, label='actual_prices', color='red')
plt.title('predicted_prices vs actual_prices')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()


