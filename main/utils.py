import sys
import time

class UTILS:
    def __init__(self) -> None:
        pass
    
    def type_response(self,response):
        for char in response:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.01)
        print()
        
    def introduce(self):
        self.type_response('Hello, I am PANDA, Paradgim-based Artificial Neural Dialogue Agent. An AI Language Model which is able to predict next sequence of words based on the input sequence of words.')