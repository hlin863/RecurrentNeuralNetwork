import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import layers, models, losses, optimizers
from keras.datasets import imdb
import matplotlib.pyplot as plt

# Load dataset
vocab_size = 5000
(X_train, _), (X_test, _) = imdb.load_data(num_words=vocab_size)

# Convert dataset to text and concatenate
word_index = imdb.get_word_index()
index_word = {value: key for key, value in word_index.items()}
all_texts = [' '.join([index_word[word] for word in review])
             for review in np.concatenate([X_train, X_test])]
text = ' '.join(all_texts)

# Tokenize and create sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
seq_length = 30
data = [sequences[i: i + seq_length + 1]
        for i in range(len(sequences) - seq_length)]
data = np.array(data)
X, y = data[:, :-1], data[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Model for text generation
model = models.Sequential()
embeddingSize = 100
model.add(layers.Embedding(vocab_size, embeddingSize, input_length=seq_length))
model.add(layers.LSTM(150, return_sequences=True))
model.add(layers.LSTM(150))
model.add(layers.Dense(vocab_size, activation='softmax'))

optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(loss=losses.CategoricalCrossentropy(),
              optimizer=optimizer, metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, batch_size=256)

# Generate text function


def generate_text(model, tokenizer, init_sentence="<OOV>", max_length=100):
    for _ in range(max_length):
        token_list = tokenizer.texts_to_sequences([init_sentence])[0]
        token_list = pad_sequences(
            [token_list], maxlen=seq_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_token = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break
        init_sentence += " " + output_word
    return init_sentence


generated_text = generate_text(
    model, tokenizer, init_sentence="love", max_length=50)
print(generated_text)
