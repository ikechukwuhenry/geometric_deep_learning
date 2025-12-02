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
cleaned_filename = "twitter_training_cleaned.csv"
absolute_cleaned_file_path = os.path.join(os.getcwd(),'nlp_datasets', cleaned_filename)

# Read the cleaned CSV file into a DataFrame
df = pd.read_csv(absolute_cleaned_file_path)
print(df.head())
print(df.columns)

df = df.sample(frac=0.1)  # Shuffle the DataFrame
print(df.head(2))

# Data preprocessing
# lower case
df['tweet_text'] = df['tweet_text'].str.lower()

# drop rows with missing values in 'sentiment' or 'tweet_text' columns
df.dropna(subset=['sentiment', 'tweet_text'], inplace=True)

print(df.isna().sum())

# remove irrelevant sentiment rows
df = df[df['sentiment'] != "Irrelevant"]


# remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

df['tweet_text'] = df['tweet_text'].apply(remove_urls)

# remove punctuations
def remove_punctuations(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

df['tweet_text'] = df['tweet_text'].apply(remove_punctuations)

# remove @ mentions
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

df['tweet_text'] = df['tweet_text'].apply(remove_mentions)

# remove ellipsis
def remove_ellipsis(text):
    return re.sub(r'\.+', ' ', text)

df['tweet_text'] = df['tweet_text'].apply(remove_ellipsis)  


# remove html
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

df['tweet_text'] = df['tweet_text'].apply(remove_html)

# remove stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    temp_text = word_tokenize(text)
    for word in temp_text:
        if word in stop_words:
            text = text.replace(word, '')
    return text


df['tweet_text'] = df['tweet_text'].apply(remove_stopwords)


from nltk.stem import PorterStemmer

def Stemming(text):
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemming_words = []
    for token in tokens:
        stemming_token = ps.stem(token)
        stemming_words.append(stemming_token)
    return ' '.join(stemming_words)


df['tweet_text'] = df['tweet_text'].apply(Stemming)
print(df.head())

# change the target values to categorical(numerical) values
df['sentiment'] = df['sentiment'].map({'Positive': 2, 'Negative': 0, 'Neutral': 1})
print(df.head())

Y = df['sentiment']

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
X = tf.fit_transform(df['tweet_text']).toarray()
print(X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.shape)

shape = X_train.shape

print(shape[1])

print(X_test.shape)

print(type(X_train))

print(type(Y_train))

Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

print(X_train.ndim)


import torch
from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())

test_set = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f'Using device: {device}')

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # get the last time step
        return out
    

input_size = shape[1]
hidden_size = 128
output_size = 3  # positive, negative, neutral
num_layers = 2
num_epochs = 10
batch_size = 64
learning_rate = 0.001

model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Add an additional dimension for sequence length
        X_batch = X_batch.unsqueeze(1)  # add sequence dimension

        outputs = model(X_batch)

        # apply softmax to get probabilities
        # outputs = torch.softmax(outputs.squeeze())


        loss = criterion(outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            X_batch = X_batch.unsqueeze(1)  # add sequence dimension

            outputs = model(X_batch)
            # _, predicted = torch.max(outputs.data, 1)
            predicted = torch.softmax(outputs, dim=1).argmax(dim=1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

        print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')