import re

def preprocess(sentence):
    sentence=str(sentence)
    filtered_words = re.sub('\W+',' ', sentence)
    
    return filtered_words