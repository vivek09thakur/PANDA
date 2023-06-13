import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


text_data = [
    "hello",
    "how are you doing",
    "what is your name",
    "where are you from",
    "hi, how are you doing?",
    "i'm fine. how about yourself?",
    "i'm pretty good. thanks for asking.",
    "no problem. so how have you been?",
    "i've been great. what about you?",
    "i've been good. i'm in school right now."
    "what school do you go to?",
    "i go to pcc.",
    "do you like it there?",
    "it's okay. it's a really big campus."
    "good luck with school.",
    "how's it going?",
    "i'm doing well. how about you?"
    "never better, thanks.",
    "so how have you been lately?",
    "i've actually been pretty good. you?",
    "i'm actually in school right now.",
    "which school do you attend?",
    "i'm attending pcc right now.",
    "are you enjoying it there?",
    "where are you going to school?",
    "i'm going to pcc.",
    "how do you like it so far?",
    "i like it so far. my classes are pretty good right now.",
    "it's an ugly day today.",
    "i know. i think it may rain.",
    "it's the middle of summer, it shouldn't rain today.",
    "that would be weird.",
    "yeah, especially since it's ninety degrees outside.",
    "i know, it would be horrible if it rained and it was hot outside.",
    "yes, it would be. ",
    "i really wish it wasn't so hot every day. ",
    "me too. i can't wait until winter.",
    "i like winter too, but sometimes it gets too cold.",
    "i'd rather be cold than hot.",
    "it doesn't look very nice outside today.",
    "you're right. i think it's going to rain later.",
    "in the middle of the summer, it shouldn't be raining.",
    "that wouldn't seem right.",
    "considering that it's over ninety degrees outside, that would be weird.",
    "exactly, it wouldn't be nice if it started raining. it's too hot.",
    "i know, you're absolutely right.",
    "i wish it would cool off one day."
]

# with open('dialogs.txt','r') as f:
#     text_data = f.readline()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
next_words = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        next_words.append(n_gram_sequence[-1])

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

predictors, label = input_sequences[:, :-1], input_sequences[:, -1]



# Build the model
model = Sequential()
model.add(Embedding(total_words, 60, input_length=max_sequence_len-1))
model.add(LSTM(500))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(predictors, label, epochs=600, verbose=2)
model.save('PANDA.h5')


def predict_next_words(seed_text, num_words=5):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
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
        token_list = np.append(token_list[:, 1:], [[predicted_index]], axis=1)

    return predicted_words


print('I\'m PANDA , Paradigm-based Artificial Neural Dialogue Agent , A Language Model which is able to predict next words')

while True:
    user_input = input("user > ")
    response = predict_next_words(user_input)
    print("next word > ", response)
