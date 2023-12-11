"""This module is for text processing for model prediction."""

import nltk
import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

count_vectorizer = joblib.load("app/count_vectorizer.pkl")

def tokenize_text(text):
    """This function tokenizes the input text"""
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    """This function is used to remove the stopwords from the text"""
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def pos_tagging(tokens):
    """This function is used to pos_tagging the text"""
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    return ' '.join(nouns)

def vectorize_text(text: list[str]):
    """This function transforms data for prediction"""
    X_vectorized = count_vectorizer.transform([text])
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(X_vectorized)
    return tfidf_matrix

def preprocessing_fn(x):
    """This function performs preprocessing including tokenization, removing stopwords, lemmatization, and extracting nouns"""
    tokens = tokenize_text(x)
    tokens_without_stopwords = remove_stopwords(tokens)
    nouns = pos_tagging(tokens_without_stopwords)
    X_vectorized = vectorize_text(nouns)
    return X_vectorized


def run_preprocessing_fn(X):
    """ This functions runs the preprocessing pipeline """
    processed_data = [preprocessing_fn(text) for text in X]
    return processed_data

#TODO: cambiar nombre de variables por text



