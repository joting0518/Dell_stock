修改程式碼，改善逆縮放的地方https://colab.research.google.com/drive/1tt6cRr7e3kLcT9I8PPe7i-0wks8HMOAN?usp=sharing
本專案利用RNN架構的LSTM去進行股價預測(DELL)，這次以預測收盤價為主，而比較特別的應用是
1. scaler進行特徵縮放
2. 給模型前60天的資訊，以矩陣的形式傳入，讓模型學習時間序列的概念，去更好預測第61天的股價（收盤價）
3. 轉換成模型學習格式（list->numpy array）
4. 建立7層模型 （3 層 LSTM，3 層 Dropout，和 1 層 Dense）
5. 前面有使用到scaler，故要將特徵逆縮放，得到原始股價資訊
6. 繪製折線圖去表示predicted_prices vs actual_prices
資料來源：kaggle https://www.kaggle.com/datasets/kalilurrahman/dell-stock-data-latest-and-updated
參考資料：暸解LSTM的設計
https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_rnns_lstm_work.html