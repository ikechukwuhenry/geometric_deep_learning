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


from preprocessing_text import preprocess_tweet

# collect data and read the file into the program
file_name = "twitter_training.csv"
absolute_file_path = os.path.join(os.getcwd(),'nlp_datasets', file_name)
tweets_file = open(absolute_file_path, "r")
tweets = tweets_file.readlines()
tweets_file.close()

positive_tweets = []
negative_tweets = []
neutral_tweets = []

# clean tweets and remove unnecessary characters
pattern = r'\.+'
repl = ' '
tweets = [re.sub(pattern, repl, tweet) for tweet in tweets]

pattern2 = r'@'
repl2 = ''
tweets = [re.sub(pattern2, repl2, tweet) for tweet in tweets]

for tweet in tweets:
    text = tweet.split(",")
    
    if text[2] == "Positive":
        positive_tweets.append(text[3])
    elif text[2] == "Negative":
        negative_tweets.append(text[3])
    elif text[2] == "Neutral":
        neutral_tweets.append(text[3])

print(f"Number of positive tweets: {len(positive_tweets)}")

processed_positive_tweets = preprocess_tweet(positive_tweets)
processed_negative_tweets = preprocess_tweet(negative_tweets)
processed_neutral_tweets = preprocess_tweet(neutral_tweets)

vocab = set(processed_positive_tweets + processed_negative_tweets + processed_neutral_tweets)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
positive_tweets_freq_dist = FreqDist(processed_positive_tweets)
negative_tweets_freq_dist = FreqDist(processed_negative_tweets)
neutral_tweets_freq_dist = FreqDist(processed_neutral_tweets)
print(f"Most common words in positive tweets: {positive_tweets_freq_dist.most_common(10)}")

total_pos_words = len(processed_positive_tweets)
total_neg_words = len(processed_negative_tweets)
total_neut_words = len(processed_neutral_tweets)

priors = {
    "Positive": len(positive_tweets) / (len(positive_tweets) + len(negative_tweets) + len(neutral_tweets)),
    "Negative": len(negative_tweets) / (len(positive_tweets) + len(negative_tweets) + len(neutral_tweets)),
    "Neutral": len(neutral_tweets) / (len(positive_tweets) + len(negative_tweets) + len(neutral_tweets)),
}

print(f"Prior probabilities: {priors}")
log_priors =  { cls: math.log(prior) for cls, prior in priors.items() }# np.log(priors)
print(f"Log prior: {log_priors}")

pos_tweets_prob = {}
neg_tweets_prob = {}
neut_tweets_prob = {}
lambda_of_word = {}

for word in vocab:
    pos_tweets_prob[word] = (positive_tweets_freq_dist[word] + 1) / (total_pos_words + vocab_size)
    neg_tweets_prob[word] = (negative_tweets_freq_dist[word] + 1) / (total_neg_words + vocab_size)
    neut_tweets_prob[word] = (neutral_tweets_freq_dist[word] + 1) / (total_neut_words + vocab_size)
    lambda_of_word[word] = math.log(pos_tweets_prob[word]) - math.log(neg_tweets_prob[word])


# print(f"Probability of 'happy' in positive tweets: {pos_tweets_prob['happy']}")
# print(f"Probability of 'happy' in negative tweets: {neg_tweets_prob['happy']}")
print(f"Probability of 'sad' in positive tweets: {pos_tweets_prob['sad']}")
print(f"Probability of 'sad' in negative tweets: {neg_tweets_prob['sad']}")
print(f"Probability of 'sad' in neutral tweets: {neut_tweets_prob['sad']}")
print(f"Lambda of 'sad': {lambda_of_word['sad']}")