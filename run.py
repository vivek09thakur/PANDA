from PANDA.panda import PANDA

DATASET_PATH = 'Colab Notebooks/PANDA/dataset/chat_dataset.txt'
MODEL_PATH = 'Colab Notebooks/PANDA/model/paradigm_based_artificial_neural_daiglog_agent.h5'
NO_OF_TOKENS = 100

panda = PANDA(DATASET_PATH, MODEL_PATH, NO_OF_TOKENS)
panda.preprocess_data()
panda.generate_pad_sequences()
panda.train_or_load_model()


# Main-loop
while True:
    prompt = input('<user> ')
    completion = panda.completion(prompt)
    print('<panda> ' + completion)