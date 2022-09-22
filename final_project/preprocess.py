import re
import csv
import string
import numpy as np
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def remove_non_utf8_characters(text):
    text = unicodedata.normalize('NFD', text)
    return re.sub(r'[^\x00-\x7f]', r'', text)
    
def remove_html_tags(text):
    # Remove html tags from a string
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_URL(text):
    # Remove URLs from a text string
    return re.sub(r"http\S+", "", text)

def convert_to_lowercase(text):
    # Convert a string to lowercase
    return text.lower()

def remove_punctuation(text):
    # Remove punctuation from a string
    return text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

def remove_digits(text):
    # Remove digits from a string
    return re.sub(r'\d+', '', text)

def lemmatize(text):
    # Lemmatize a string
    lmtzr = WordNetLemmatizer()
    lemmatized_text = [lmtzr.lemmatize(word) for word in text]
    return lemmatized_text

def remove_single_characters(text):
    # Remove single characters from a string
    return re.sub(r"\b\w\b", "", text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word not in stop_words]

# filtered_sentence = remove_stopwords(text)
# https://stackabuse.com/removing-stop-words-from-strings-in-python/


def denoise_text(text):
    # Remove stopwords, punctuation, and convert to lowercase etc
    text = remove_non_utf8_characters(text)
    text = remove_html_tags(text)
    text = remove_URL(text)
    text = remove_punctuation(text)
    text = remove_single_characters(text)
    text = remove_digits(text)
    text = convert_to_lowercase(text)
    text = word_tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

def convert_string_to_true_false(text):
    # Convert a string to True or False
    if text == 'positive':
        return True
    else:
        return False

def read_and_preprocess():
    df = pd.read_csv('IMDB Dataset.csv', encoding = 'utf-8')
    df['review'] = df['review'].apply(denoise_text)
    df['sentiment'] = df['sentiment'].apply(convert_string_to_true_false)
    X = df['review'].values
    Y = df['sentiment'].values
    return X, Y

X, Y = read_and_preprocess()  
with open('X_Y.npy', 'wb') as f:
    np.save(f, X)
    np.save(f, Y)


