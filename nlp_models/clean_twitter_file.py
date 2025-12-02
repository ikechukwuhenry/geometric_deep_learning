import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer,TreebankWordDetokenizer
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import numpy as np
import torch
import os
import re
import math
import pandas as pd

# collect data and read the file into the program
file_name = "twitter_training.csv"
absolute_file_path = os.path.join(os.getcwd(),'nlp_datasets', file_name)

df = pd.read_csv(absolute_file_path)
print(df.head())
print(df.columns)
df.columns= ['id', 'boardlands', 'sentiment', 'tweet_text']
print(df.columns)
print(df.head())

df.drop(columns=['id', 'boardlands'], inplace=True)
print(df.head())
cleaned_filename = "twitter_training_cleaned.csv"
absolute_cleaned_file_path = os.path.join(os.getcwd(),'nlp_datasets', cleaned_filename)
df.to_csv(absolute_cleaned_file_path, index=False)