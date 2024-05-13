from main.transform import PANDA
from main.train import TRAINER
import argparse
    
argparser = argparse.ArgumentParser()
argparser.add_argument('--train', action='store_true')
argparser.add_argument('--infer', action='store_true')
argparser.add_argument('--train_prompts', type=str)
argparser.add_argument('--epochs', type=int)
argparser.add_argument('--prompt_file', type=str)
argparser.add_argument('--pretrained_model', type=str)
argparser.add_argument('--infer_rounds', type=int)
args = argparser.parse_args()

if args.train:
    trainer = TRAINER(args.train_prompts)
    trainer.build_model(args.epochs)
    trainer.build_model(epochs=args.epochs)

elif args.infer:
    
    model = PANDA([args.prompt_file, 50])
    model.pretrained(args.pretrained_model)
    
    for i in range(args.infer_rounds):
        user_input = input('=>')
        print(model.infer(user_input))
