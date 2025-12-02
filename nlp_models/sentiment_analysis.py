import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer,TreebankWordDetokenizer
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import numpy as np
import torch


# collect data and read the file into the program
tweets_file_path = "/Users/ikechukwumichael/Desktop/geometric_deep_learning/nlp_datasets/twitter_training.csv"
tweets_file = open(tweets_file_path, "r")
tweets = tweets_file.readlines()
tweets_file.close()
# print(f"Number of tweets: {len(tweets)}")


import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(tweets_file_path)
# Display the first few rows of the DataFrame
# print(df.head())

# preprocess the tweets

# group tweets by sentiment
# tweets = tweets.lower()

positive_tweets = []
negative_tweets = []
neutral_tweets = []
for tweet in tweets:
    text = tweet.split(",")
    # preprocessed_text = preprocess_tweet(text)
    
    if text[2] == "Positive":
        positive_tweets.append(text[3])
    elif text[2] == "Negative":
        negative_tweets.append(text[3])
    elif text[2] == "Neutral":
        neutral_tweets.append(text[3])

print(f"Number of positive tweets: {len(positive_tweets)}")
print(f"Number of negative tweets: {len(negative_tweets)}")
print(f"Number of neutral tweets: {len(neutral_tweets)}")


# tokenize the tweets and create the vocabulary
tokenizer = WordPunctTokenizer()
positive_tweets = [tokenizer.tokenize(tweet.lower()) for tweet in positive_tweets]
negative_tweets = [tokenizer.tokenize(tweet.lower()) for tweet in negative_tweets]
neutral_tweets = [tokenizer.tokenize(tweet.lower()) for tweet in neutral_tweets]
print(f"Tokenized positive tweets: {positive_tweets[:2]}")
# print(f"Tokenized negative tweets: {negative_tweets[:2]}")
# print(f"Tokenized neutral tweets: {neutral_tweets[:2]}")

# remove stop words and punctuation
stop_words = set(stopwords.words("english"))
# punctuation = set(string.punctuation)
# positive_tweets = [tweet for tweet in positive_tweets if tweet not in stop_words]
positive_tweets = [[word for word in tweet if word not in stop_words] for tweet in positive_tweets]
positive_tweets = [[word for word in tweet if word not in string.punctuation] for tweet in positive_tweets]
print(f"Positive tweets after stop word removal: {positive_tweets[:5]}")
portstemmer = PorterStemmer()
positive_tweets = [[portstemmer.stem(word) for word in tweet] for tweet in positive_tweets]
print(f"Positive tweets after stemming: {positive_tweets[:5]}")
# positive_tweets = 
flattened_positive_tweets = []
for tweet in positive_tweets:
    flattened_positive_tweets.extend(tweet)

# print(f"flattened positive tweets: {flattened_positive_tweets[:10]}")

# process negative tweets
negative_tweets = [[word for word in tweet if word not in stop_words] for tweet in negative_tweets]
negative_tweets = [[word for word in tweet if word not in string.punctuation] for tweet in negative_tweets]
print(f"Negative tweets after stop word removal: {negative_tweets[:5]}")
portstemmer = PorterStemmer()
negative_tweets = [[portstemmer.stem(word) for word in tweet] for tweet in negative_tweets]
print(f"Positive tweets after stemming: {negative_tweets[:5]}")
# positive_tweets = 
flattened_negative_tweets = []
for tweet in negative_tweets:
    flattened_negative_tweets.extend(tweet)


print(f"Negative tweets after processing: {flattened_negative_tweets[:5]}")

# process neutral tweets
neutral_tweets = [[word for word in tweet if word not in stop_words] for tweet in neutral_tweets]
neutral_tweets = [[word for word in tweet if word not in string.punctuation] for tweet in neutral_tweets]
print(f"Neutral tweets after stop word removal: {neutral_tweets[:5]}")
portstemmer = PorterStemmer()
neutral_tweets = [[portstemmer.stem(word) for word in tweet] for tweet in neutral_tweets]
# print(f"Neutral tweets after stemming: {neutral_tweets[:2]})
# positive_tweets = 
flattened_neutral_tweets = []
for tweet in neutral_tweets:
    flattened_neutral_tweets.extend(tweet)

print(f"Neutral tweets after processing: {flattened_neutral_tweets[:5]}")


# print(f"Neutral tweets after processing: {neutral_tweets[:2]}")

# create the vocabulary
vocab = flattened_positive_tweets + flattened_negative_tweets + flattened_neutral_tweets
print(f"Vocabulary size: {len(vocab)}")
vocab = set(vocab)
vocab_size = len(vocab)
print(f"Unique vocabulary size: {vocab_size}")
