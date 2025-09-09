import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
values = data['demand'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# Prepare dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 24  # e.g., last 24 hours
X, y = create_dataset(scaled_values, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=20, batch_size=32, validation_data=(testX, testY))

# Forecast
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse transform predictions
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
trainY = scaler.inverse_transform([trainY])
testY = scaler.inverse_transform([testY])

# Plot results
plt.figure(figsize=(10,6))
plt.plot(data.index[look_back+1:train_size+look_back+1], trainY.flatten(), label='Train Actual')
plt.plot(data.index[look_back+1:train_size+look_back+1], trainPredict.flatten(), label='Train Predict')
plt.plot(data.index[train_size+look_back+1:], testY.flatten(), label='Test Actual')
plt.plot(data.index[train_size+look_back+1:], testPredict.flatten(), label='Test Predict')
plt.xlabel('Time')
plt.ylabel('Energy Demand')
plt.title('Renewable Energy Demand Forecasting')
plt.legend()
plt.show()
