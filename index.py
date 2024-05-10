from main.transform import PANDA

model = PANDA(['Train/Prompts/train_prompts.txt', 50])
model.pretrained('saved_model/panda-25k-2.5-lstm-lm')

for i in range(5): 
    user_prompt = input('=> ')
    completion = model.infer(user_input=user_prompt)
    print("=> ",completion)