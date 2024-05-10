from keras.models import load_model
import numpy as np
from .train import TRAINER


class PANDA(TRAINER):
    def __init__(self,parameters:list):
        TRAINER.__init__(self,parameters[0])
        self.tokens = parameters[1]
        
        if len(parameters) > 2:
            raise RuntimeError(f"\nmax params count exceeded (max params count = 2) given {len(parameters)}\n")
        
    def pretrained(self,model_path):
        self.model = load_model(model_path)    
            
    def infer(self,user_input):
        token_list = self.process_user_prompt(user_prompt=user_input)
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