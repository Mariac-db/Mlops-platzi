"""This module is for text processing for model prediction."""

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def tokenize_text(text):
    """This function tokenizes the input text"""
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    """This function is used to remove the stopwords from the text"""
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def lemmatize(tokens):
    """This function is used to lemmatize the text""" 
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens

def pos_tagging(tokens):
    """This function is used to pos_tagging the text"""
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    return ' '.join(nouns)

def preprocessing_fn(x: str):
    """This function performs preprocessing including tokenization, removing stopwords, lemmatization, and extracting nouns"""
    tokens = tokenize_text(x)
    tokens_without_stopwords = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize(tokens_without_stopwords)
    nouns = pos_tagging(lemmatized_tokens)
    return nouns

