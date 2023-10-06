import os
import sys
import time
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


class PANDA:
    
    def __init__(self,prompts,model_name,tokens=25):
        self.prompts = prompts
        self.tokens = tokens 
        with open(self.prompts,'r') as f:
            # Read the lines from the prompts file
            self.text_data = f.readlines()
        # Create a tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.text_data)
        self.total_words = len(self.tokenizer.word_index) + 1
        self.model_name = model_name
        # self.max_sequence_len = 0
        
    
    def preprocess_data(self):
        self.input_sequences = []
        self.next_words = []
        for line in self.text_data:
            # Convert the text to sequences
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1,len(token_list)):
                # Create n-grams
                n_grams = token_list[:i+1]
                self.input_sequences.append(n_grams)
                self.next_words.append(token_list[i])
                
    def generate_pad_sequences(self):
        # Pad sequences
        self.max_sequence_len = max([len(x) for x in self.input_sequences])
        self.input_sequences = np.array(
            pad_sequences(self.input_sequences,
                          maxlen=self.max_sequence_len,padding='pre'))
        self.predictors, self.label = self.input_sequences[:, :-1], self.input_sequences[:, -1]
        
    def create_model(self):
        # Create model
        self.model = Sequential()
        self.model.add(Embedding(self.total_words,100,
                                 input_length=self.max_sequence_len-1))
        self.model.add(LSTM(500))
        self.model.add(Dense(self.total_words,activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',metrics=['accuracy'])
        self.model.fit(self.predictors,self.label,epochs=500,
                       verbose=1)
        self.model.save(self.model_name)
        
    def load_model(self):
        self.model = load_model(self.model_name)
        
    def train_or_load_model(self):
        if os.path.exists(self.model_name):
            self.load_model()
        else:
            self.create_model()
            
    def completion(self,user_input):
        # Predictions
        token_list = self.tokenizer.texts_to_sequences([user_input])[0]
        token_list = pad_sequences([token_list],
                                      maxlen=self.max_sequence_len-1,
                                      padding='pre')
        predicted_words = []
        
        for _ in range(self.tokens):
            predicted = self.model.predict(token_list,verbose=0)
            predicted_index = np.argmax(predicted)
            output_word = ''
            for word,index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            predicted_words.append(output_word)
            token_list = np.append(
                token_list[:,1:],
                [[predicted_index]],
                axis=1)
        return ' '.join(predicted_words)
    
    def type_response(self,response):
        for char in response:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01)
        print()
        
    def introduce(self):
        self.type_response('Hello, I am PANDA, Paradgim-based Artificial Neural Dialogue Agent. An AI Language Model which is able to predict next sequence of words based on the input sequence of words.')