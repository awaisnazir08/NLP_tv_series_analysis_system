import re

# Remove actions from transcript
def remove_parenthesis(text):
    result = re.sub(r'\(.*?\)', '', text)
    return result