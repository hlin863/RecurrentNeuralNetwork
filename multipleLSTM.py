import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam

# Load and process the text data
training_file = '2600-0.txt'
raw_text = open(training_file, 'r', encoding='utf-8').read().lower()

chars = sorted(list(set(raw_text)))
nVocabs = len(chars)
seqLength = 100  # Reduced sequence length
hiddenUnits = 700

indexToChars = dict((i, c) for i, c in enumerate(chars))
charsToIndex = dict((c, i) for i, c in enumerate(chars))

# Convert chars to integers
raw_int = [charsToIndex[c] for c in raw_text]

# Create a tf.data dataset from raw_text
char_dataset = tf.data.Dataset.from_tensor_slices(raw_int)

# Convert char_dataset to sequences
sequences = char_dataset.batch(seqLength+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(10000).batch(
    128, drop_remainder=True)  # Increased batch size

# Construct the LSTM model
model = Sequential([
    Embedding(nVocabs, 256, input_length=seqLength),
    LSTM(hiddenUnits, return_sequences=True),
    Dense(nVocabs, activation='softmax')
])

# Using Gradient Clipping
optimizer = Adam(lr=0.002, clipnorm=1.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

# Callbacks
filePath = 'model.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filePath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', mode='min', verbose=1, patience=3)


class ResultChecker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        index = np.random.randint(nVocabs)
        yChar = [indexToChars[index]]
        x = np.zeros((1, seqLength))
        for i in range(seqLength):
            x[0, i] = index
            index = np.argmax(self.model.predict(x)[0][-1])
            yChar.append(indexToChars[index])
        print(''.join(yChar))


resultChecker = ResultChecker()

model.fit(dataset, epochs=300, callbacks=[
          checkpoint, earlyStop, resultChecker])
