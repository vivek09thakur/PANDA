from .data_processor import DATA_PROCESSOR
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

class TRAINER(DATA_PROCESSOR):
    
    def __init__(self,prompts) -> None:
        DATA_PROCESSOR.__init__(self,prompts)
        DATA_PROCESSOR.process_prompt_data()
        DATA_PROCESSOR.generate_padded_seqs()
        
    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.total_words,
                                 100,
                                 input_length=self.max_seq_len-1))
        self.model.add(LSTM(1000))
        self.model.add(LSTM(500))
        self.model.add(LSTM(250))
        self.model.add(LSTM(125))
        self.model.add(LSTM(100))
        self.model.add(LSTM(1000))
        self.model.add(Dense(self.total_words,activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(self.predictors,
                       self.labels,
                       2000,
                       verbose=2)
        self.model.save("panda-25k-2.5-lstm-lm")