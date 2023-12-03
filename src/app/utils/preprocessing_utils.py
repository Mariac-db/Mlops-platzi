"""This module is for text processing for model prediction."""

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

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

def preprocessing_fn(x: str):
    """This function performs preprocessing including tokenization, removing stopwords, lemmatization, and extracting nouns"""
    tokens = tokenize_text(x)
    tokens_without_stopwords = remove_stopwords(tokens)
    nouns = pos_tagging(tokens_without_stopwords)
    return nouns


#print(preprocessing_fn("Hi i wanna kwnow about invertions and how can I get a loan for a house"))