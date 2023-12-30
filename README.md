## Paradigm-based Artificial Neural Dailogue Agent or PANDA
> A LSTM AI Language Model trained on 25K Tokens of English Language and C-Progamming Language which can predict next sequence of words and it can also write some basic codes to some extent.

**PANDA (Paradigm-based Artificial Neural Dialogue Agent)** is an implementation of a Language Model (LM) using Keras and LSTM (Long Short-Term Memory) neural networks. This LM is designed for text generation based on a provided set of prompts. 

It combines data preprocessing, neural network architecture, training, and text generation features to create a versatile LM capable of generating text based on input prompts. These features make it a valuable tool for various natural language processing tasks and creative text generation applications.

## Usage

In Repository, there are two files located at `Colab Notebook` folder :
- `PANDA.ipynb` : This is the main experiment file which contains all the code for training and testing the model. You can run this file on Google Colab.
- `Panda_Code_Refactored.ipynb` : This file contains the code for main refactored code for the model. You can run this file on Google Colab.

Also, You can use this model by importing `PANDA.py` file in your project and use it as a class. Training this model on your own local machine takes a lot of time and resources. So,Therefor we have a *saved model* of it as `PANDA.h5` you can find it at `Saved Model/panda.h5` or if you want to train it by your self then I recommend you to use Google Colab for training this model.

Spinnet of code for using PANDA class in your project:

```python
from PANDA.panda import PANDA

DATASET = 'Colab Notebook/dataset/prompt_completion.txt',
MODEL_PATH = 'Saved Model/panda.h5', 
TOKENS = 50 


panda = PANDA(DATASET,MODEL_PATH,TOKENS)
panda.preprocess_data()
panda.generate_pad_sequences()
panda.train_or_load_model(neuron_num=1000)

panda.introduce()
if __name__=='__main__':
    while True:
        prompts = input(f"\n ↳ (user) : " )
        completion = panda.completion(prompts)
        panda.type_response(completion)
```

### Contributing

Contributions are welcome! For bug reports or requests please submit an issue. For new feature contribution please submit a pull request. If you would like to contribute to the project, here are some ways you can help:

- Improve the documentation
- Report bugs
- Fix bugs and submit pull requests
- Write, clarify, or fix documentation
- Suggest or add new features