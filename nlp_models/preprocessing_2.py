# pip install nltk
# pip intall emoji

import nltk
from nltk.tokenize import word_tokenize
import emoji
import re

nltk.download('punkt')

corpus = 'Who love "word embeddings" in 2020? I do!!!"'

data  = re.sub(r'[,!?;-]+', '.', corpus)
print(data)
data = nltk.word_tokenize(data)
print(data)
data = [ ch.lower() for ch in data 
        if ch.isalpha()
        or ch == '.'
        # or emoji.get_emoji_regexp().search(ch)
        ]