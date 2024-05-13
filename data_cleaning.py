# path: train/data_cleaning.py
# This script only designed for the dataset I used to train this language model. If you are using a different dataset you have to clean up dataset by yourself.

# training_data = "train/prompts/train_prompts.txt"
# cleaned_data = "train/prompts/cleaned_prompts.txt"

# cleaned_lines = []

# with open(training_data, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# junk_words = ['Human 1:', 'Human 2:','?','!','.','\n'] 
# for line in lines:
#     cleaned_line = line
#     for junk_word in junk_words:
#         if junk_word in cleaned_line: 
#             cleaned_line = cleaned_line.replace(junk_word, '')  
#     cleaned_lines.append(cleaned_line.strip())

# with open(cleaned_data, 'w', encoding='utf-8') as f2:
#     for cleaned_line in cleaned_lines:
#         f2.write(cleaned_line + '\n')



training_data = "train/prompts/train_prompts.txt"
cleaned_data = "train/prompts/cleaned_prompts.txt"

cleaned_lines = []

with open(training_data, 'r', encoding='utf-8') as f:
    lines = f.readlines()

junk_words = ['Human 1:', 'Human 2:', '?', '!', '.', '\n']
for i in range(0, len(lines), 2): 
    cleaned_line1 = lines[i]
    cleaned_line2 = lines[i + 1] if i + 1 < len(lines) else ''

    for junk_word in junk_words:
        if junk_word in cleaned_line1:
            cleaned_line1 = cleaned_line1.replace(junk_word, '')
        if junk_word in cleaned_line2:
            cleaned_line2 = cleaned_line2.replace(junk_word, '')

    combined_line = cleaned_line1.strip() + ' ' + cleaned_line2.strip()  
    cleaned_lines.append(combined_line)

with open(cleaned_data, 'w', encoding='utf-8') as f2:
    for cleaned_line in cleaned_lines:
        f2.write(cleaned_line + '\n')
