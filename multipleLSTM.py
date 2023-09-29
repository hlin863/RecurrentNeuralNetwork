"""
LSTM Text Generator

This script reads a text file and uses it to train a two-layer LSTM model to generate text sequences.

Dependencies:
    - numpy
    - tensorflow
    - keras

Workflow:
1. Load and preprocess the input text file.
2. Prepare the dataset for sequence prediction.
3. Define the LSTM model.
4. Compile and display the model summary.

Modules:
    - numpy: Used for numerical operations.
    - tensorflow: Open source machine learning framework.
    - keras: High-level neural networks API, running on top of TensorFlow.

Steps:
1. Import necessary libraries.
2. Load the input text file and convert it to lowercase.
3. Calculate and print statistics about the text (e.g., number of characters, vocabularies).
4. Prepare sequences of characters to be fed into the LSTM.
5. Define the LSTM model architecture.
6. Compile the model.
7. Print the model summary.
"""

import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Path to the input text file.
training_file = '2600-0.txt'

# Read and load the content of the text file.
raw_text = open(training_file, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
print(raw_text[:200])

# Calculate number of characters and unique characters.
nChars = len(raw_text)
chars = sorted(list(set(raw_text)))
nVocabs = len(chars)
print("The total number of vocabularies is: ", nVocabs)

# Create mappings from index to character and character to index.
indexToChars = dict((i, c) for i, c in enumerate(chars))
charsToIndex = dict((c, i) for i, c in enumerate(chars))

# Define the length of each sequence.
seqLength = 160
nSeq = int(np.floor((nChars - 1) / seqLength))
print("The total number of sequences is: ", nSeq)

# Prepare input and target sequences for the LSTM.
X = np.zeros((nSeq, seqLength, nVocabs))
y = np.zeros((nSeq, seqLength, nVocabs))
for i in range(nSeq):
    xSequence = raw_text[i * seqLength: (i + 1) * seqLength]
    xSequenceIndex = [charsToIndex[char] for char in xSequence]
    inputSequence = np.zeros((seqLength, nVocabs))
    for j in range(seqLength):
        inputSequence[j][xSequenceIndex[j]] = 1
    X[i] = inputSequence
    ySequence = raw_text[i * seqLength + 1: (i + 1) * seqLength + 1]
    ySequenceIndex = [charsToIndex[char] for char in ySequence]
    targetSequence = np.zeros((seqLength, nVocabs))
    for j in range(seqLength):
        targetSequence[j][ySequenceIndex[j]] = 1
    y[i] = targetSequence

# Seed for reproducibility.
tensorflow.random.set_seed(42)

# Define hyperparameters.
hiddenUnits = 700
dropOut = 0.4
batchSize = 100
nEpoch = 300

# Define the LSTM model architecture.
model = Sequential()
model.add(LSTM(hiddenUnits, input_shape=(None, nVocabs),
          return_sequences=True, dropout=dropOut))
model.add(LSTM(hiddenUnits, return_sequences=True, dropout=dropOut))
model.add(Dense(nVocabs, activation='softmax'))

# Compile the model.
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Print model summary.
print(model.summary())
