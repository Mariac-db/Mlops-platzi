"""This module is for text processing for model prediction."""

import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime

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

def pos_tagging(tokens):
    """This function is used to pos_tagging the text"""
    tagged = pos_tag(tokens)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    return ' '.join(nouns)

def vectorize_text(text: list[str]):
    """This function transforms data for prediction"""
    # artifact with data count vectorized (pretrained)
    count_vectorizer = joblib.load("app/count_vectorizer.pkl")
    # must be a list
    X_vectorized = count_vectorizer.fit_transform([text])
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(X_vectorized)
    return tfidf_matrix

def preprocessing_fn(x):
    """This function performs preprocessing including tokenization, removing stopwords, lemmatization, and extracting nouns"""
    tokens = tokenize_text(x)
    tokens_without_stopwords = remove_stopwords(tokens)
    nouns = pos_tagging(tokens_without_stopwords)
    X_vectorized = vectorize_text(nouns)
    return X_vectorized.toarray()


def run_preprocessing_fn(X):
    # testing with dict
    initial_time = datetime.now()
    if isinstance(X, list):
        processed_data = [preprocessing_fn(text) for text in X]
        print(processed_data)
        elapsed_time = datetime.now() - initial_time
        print("time: ", elapsed_time )
        return processed_data


# if __name__ == "__main__":
        # pruebita 
#     print(test_with_list_comp(["I'm experiencing issues with my mortgage payments. The bank denied my loan application due to credit history. I need financial assistance to buy a new property. My credit score dropped after the loan default.",
#     "The bank increased the interest rates on my mortgage. I'm struggling to pay off my student loans. I'm facing foreclosure on my house. I want to refinance my home loan for better rates. The bank's customer service is very unhelpful.",
#     "I'm concerned about hidden fees on my credit card. I received a notice of overdraft fees from the bank. The mortgage terms are too strict for my situation. I need a personal loan for medical emergencies. I'm worried about identity theft and bank fraud.",
#     "My car loan interest rates are too high. I'm considering a home equity line of credit. I want to dispute some charges on my bank statement. The bank didn't notify me about the account closure. I'm seeking advice on debt consolidation options.",
#     "I'm looking for ways to improve my credit score. I'm facing financial hardships after a job loss. The loan approval process is taking too long. I'm unhappy with the bank's investment options. I want to cancel my credit card due to high fees.",
#     "I need guidance on mortgage refinancing. I'm dissatisfied with the loan repayment terms. The bank is not providing adequate insurance services. I'm considering bankruptcy due to financial struggles. I'm having trouble making ends meet.",
#     "I'm seeking a second opinion on loan modification. The bank made an error in calculating interest rates. I'm worried about rising inflation affecting loans. I want to explore home loan modification options. I received a notice of change in mortgage servicer.",
#     "I'm looking for assistance in managing debts. The bank's foreclosure process is stressful. I'm considering a short sale for my property. I'm unhappy with the terms of my business loan. I'm having difficulty accessing my online banking.",
#     "I'm concerned about mortgage insurance costs. I'm exploring options for loan deferment. The bank's handling of loan payments is inefficient. I want to renegotiate my home loan terms. I'm worried about loan affordability in the future.",
#     "I'm facing difficulties with loan interest calculations. I need clarification on mortgage forbearance terms. The bank's overdraft fees are excessive. I'm struggling to pay off multiple loans."]))
    


