import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class DATA_PROCESSOR:
    def __init__(self,prompt_data) -> None:
        self.prompt_data = prompt_data
        with open(self.prompt_data,'r') as f1:
            self.text_data = f1.readlines()
        self.tokenizer = Tokenizer()
        pass
    
    def process_raw_text(self):
        self.tokenizer.fit_on_texts(self.text_data)
        self.total_words = len(self.tokenizer.word_index) + 1
    
    def process_prompt_data(self):
        self.input_seqs = []
        self.next_words = []
        for line in self.text_data:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1,len(token_list)):
                n_grams = token_list[:i+1]
                self.input_seqs.append(n_grams)
                self.next_words.append(token_list[i])
                
    def generate_padded_seqs(self):
        self.max_seq_len = max([len(x)] for x in self.input_seqs)
        self.input_seqs = np.array(
            pad_sequences(self.input_seqs,
                          maxlen=self.max_seq_len,padding='pre'))
        self.predictors, self.label = self.input_seqs[:, :-1], self.input_seqs[:, -1]
        
    def process_user_prompt(self,user_prompt):
        token_list = self.tokenizer.texts_to_sequences([user_prompt])[0]
        token_list = pad_sequences([token_list],
                                   maxlen=self.max_seq_len-1,
                                   padding='pre')
        return token_list
                