import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

with open('dialogs.txt', 'r') as f:
    text_data = f.readlines()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
next_words = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence[:-1])
        next_words.append(n_gram_sequence[-1])

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                                        maxlen=max_sequence_len,
                                        padding='pre')
                                    )

predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

model_filename = 'PANDA.h5'
if not os.path.exists(model_filename):
    # Build the model
    model = Sequential()
    model.add(Embedding(
        total_words, 60, 
        input_length=max_sequence_len - 1)
        )
    model.add(LSTM(200))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
        )
    model.fit(predictors, 
              label, 
              epochs=200, 
              verbose=1)
    model.save(model_filename)
else:
    # Build the model
    model = Sequential()
    model.add(Embedding(
        total_words, 60, 
        input_length=max_sequence_len - 1)
        )
    model.add(LSTM(200))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
        )
    model.load_weights(model_filename)

def predict_next_words(seed_text, num_words=5):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list], 
        maxlen=max_sequence_len - 1, 
        padding='pre')
    predicted_words = []

    for _ in range(num_words):
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break
        predicted_words.append(predicted_word)
        token_list = np.append(token_list[:, 1:],
                                [[predicted_index]],
                                  axis=1)

    return predicted_words

while True:
    user_input = input("user > ")
    response = predict_next_words(user_input)
    print("next word > ", response)