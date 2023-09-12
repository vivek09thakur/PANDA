from PANDA.panda import PANDA

parameters = [
    'Colab Notebooks/dataset/prompts.txt', # prompts file
    'Saved Model/panda.h5', # model name
     25 # number of tokens to generate
]

panda = PANDA(parameters[0],parameters[1],parameters[2])
panda.preprocess_data()

if __name__=='__main__':
    
    panda.train_or_load_model()
    panda.introduce()
    
    while True:
        prompts = input(f"\n ↳ (user) : " )
        completion = panda.completion(prompts)
        panda.type_response(completion)