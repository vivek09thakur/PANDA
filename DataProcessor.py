# # Read the content of the file
# with open("dialogs.txt", "r") as file:
#     content = file.read()

# # Remove any leading or trailing whitespace
# content = content.strip()

# # Split the content into individual lines
# lines = content.split("\n")

# # Initialize variables for formatted lines
# formatted_lines = []

# # Combine the lines in the desired format
# for i in range(0, len(lines), 2):
#     if i + 1 < len(lines):
#         formatted_lines.append(lines[i] + " " + lines[i + 1])

# # Join the formatted lines into a single string
# new_content = "\n".join(formatted_lines)

# # Write the new content to a new file
# with open("new_data.txt", "w") as file:
#     file.write(new_content)

count = 0;  
   
#Opens a file in read mode  
file = open("prompts.txt", "r")  
      
#Gets each line till end of file is reached  
for line in file:  
    #Splits each line into words  
    words = line.split(" ");  
    #Counts each word  
    count = count + len(words);  
   
print("Number of tokens : " + str(count));  
file.close();
