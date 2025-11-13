# Version 1: Simple keyword search
# This is NOT fancy AI - just basic Python

# reading the file
with open("myProfile.txt", "r") as f:
    profile = f.read()

# asking a question
ques = "what is my address?"

# super simple search
if "address" in ques:
    for line in profile.split("\n"):
        if "Address" in line:
            print("Found: ", line)
else:
    print("Don't know how to answer that yet")