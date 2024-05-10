from main.transform import PANDA

model = PANDA(['Train/Prompts/train_prompts.txt', 50])
model.pretrained('saved_model/panda-25k-2.5-lstm-lm')

for i in range(5): # inference for 5 turns
    user_prompt = input('==> ')
    print(model.infer(user_input=user_prompt))