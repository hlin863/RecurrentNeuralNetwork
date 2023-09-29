import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate sample data
# For simplicity, we'll use sine wave data as a time series example
timesteps = np.linspace(0, 10*np.pi, 1000)  # Time steps
data = np.sin(timesteps)  # Sine wave data

# Prepare data for LSTM
sequence_length = 10
X, y = [], []
for i in range(len(data) - sequence_length):
    seq = data[i:i + sequence_length]
    label = data[i + sequence_length]
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Reshape the input data into [samples, timesteps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Prediction (using the trained model to forecast the next point)
test_input = X[-1]  # Take the last sequence from our dataset
test_input = test_input.reshape(1, sequence_length, 1)
predicted_output = model.predict(test_input)[0][0]
print(f"Predicted output: {predicted_output}")
