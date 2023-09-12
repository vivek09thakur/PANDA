# with open("dialogs.txt", "r") as file:
#     content = file.read()

# content = content.strip()
# lines = content.split("\n")
# formatted_lines = []

# for i in range(0, len(lines), 2):
#     if i + 1 < len(lines):
#         formatted_lines.append(lines[i] + " " + lines[i + 1])

# new_content = "\n".join(formatted_lines)

# with open("new_data.txt", "w") as file:
#     file.write(new_content)

count = 0;  
   
file = open("prompts.txt", "r")  
      
for line in file:  

    words = line.split(" ");  
    count = count + len(words);  
   
print("Number of tokens : " + str(count));  
file.close();
