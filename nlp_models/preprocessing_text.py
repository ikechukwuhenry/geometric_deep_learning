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


stopwords = set(stopwords.words("english"))


def preprocess_tweet(tweets: str) -> str:
    tokenizer = WordPunctTokenizer()

    tweets = [tokenizer.tokenize(tweet.lower()) for tweet in tweets]
    flattened_tweets = [word for tweet in tweets for word in tweet]
    # print(f"Tokenized tweets: {flattened_tweets[:12]}")

    # remove stop words and punctuation
    
    filtered_tweets = [word for word in flattened_tweets if word not in stopwords and word not in string.punctuation]

    port_stemmer = PorterStemmer()
    stemmed_tweets = [port_stemmer.stem(word) for word in filtered_tweets]
    # print(f"Stemmed tweets: {stemmed_tweets[:12]}")
    return stemmed_tweets



# example usage:
if __name__ == "__main__":
    pass