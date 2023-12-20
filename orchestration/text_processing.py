import json
import os
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
import logging
import warnings
import datetime

warnings.filterwarnings("ignore")


class TextProcessing:
    """This class is used to process the text,
    contains methods to tokenize, remove stopwords, lemmatize and pos_tagging the text
    then, this data transformed to a dataframe and saved to a CSV file
    The idea is to use this class in the pipeline to feature extration process"""

    def __init__(self, language: str):
        """This class is used to process the text
        Class parameters:
            lenguage (str): Language of the text to process
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        nltk.download("averaged_perceptron_tagger")
        self.language = language
        self.stop_words = set(stopwords.words(self.language))
        self.stemmer = SnowballStemmer(self.language)

    def tokenize(self, text: str):
        """This method is used to tokenize the text"""
        tokens = word_tokenize(text.lower(), language=self.language)
        return tokens

    def remove_stopwords(self, tokens: list):
        """This method is used to remove the stopwords from the text"""
        filtered_tokens = [
            word for word in tokens if word.lower() not in self.stop_words
        ]
        return filtered_tokens

    def lemmatize(self, tokens: list):
        """This method is used to lemmatize the text"""
        lemmatized_tokens = [self.stemmer.stem(word) for word in tokens]
        return lemmatized_tokens

    def pos_tagging(self, tokens: list):
        """This method is used to pos_tagging the text"""
        tagged = pos_tag(tokens)
        nouns = [word for word, pos in tagged if pos == "NN"]
        return " ".join(nouns)

    def text_preprocessing(self, column_to_process: pd.Series):
        """This method is used to run the whole process of cleaning the text"""
        initial_time = datetime.datetime.now()
        tokenized_text = column_to_process.apply(self.tokenize)
        text_without_stopwords = tokenized_text.apply(self.remove_stopwords)
        text_lemma = text_without_stopwords.apply(self.lemmatize)
        pos_tagging_tokens = text_lemma.apply(self.pos_tagging)
        final_time = datetime.datetime.now()
        self.logger.info(f"Text successfully processed")
        self.logger.info(f"time = {final_time - initial_time}")
        return pos_tagging_tokens  # text_without_stopwords.apply(lambda x: ' '.join(x))

    def save_processed_data(self, df: pd.DataFrame, path: str, file_name: str) -> None:
        """This method saves the processed data and labels to a CSV"""
        file_path = os.path.join(path, file_name)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Data successfully saved to {file_path}")

    def read_json(self, path: str, file_name: str):
        """This method is used to read the json file"""
        file_path = os.path.join(path, file_name)
        with open(file_path, "r") as file:
            datos = json.load(file)
        df_tickets = pd.json_normalize(datos)
        return df_tickets

    def read_csv(self, path: str, file_name: str):
        """This method is used to read the csv file"""
        file_path = os.path.join(path, file_name)
        df_tickets = pd.read_csv(file_path)
        return df_tickets

    def data_transform(self, df: pd.DataFrame):
        """This method is used to transform the data to a vector"""
        df = df[
            [
                "_source.complaint_what_happened",
                "_source.product",
                "_source.sub_product",
            ]
        ]
        df = df.rename(
            columns={
                "_source.complaint_what_happened": "complaint_what_happened",
                "_source.product": "category",
                "_source.sub_product": "sub_product",
            }
        )
        df["ticket_classification"] = (
            df["category"] + " + " + df["sub_product"]
        )
        df = df.drop(["sub_product", "category"], axis=1)
        df["complaint_what_happened"] = df["complaint_what_happened"].replace(
            "", np.nan
        )
        df = df.dropna(subset=["complaint_what_happened", "ticket_classification"])
        df = df.reset_index(drop=True)
        self.logger.info("Data successfully transformed")
        return df


    def run(self, file_name: str, version: int):
        """Runs the entire text processing pipeline."""
        name_data_input = f"{file_name}"
        PATH_DATA_RAW = "/Users/mdurango/Proyect/Mlops-platzi/orchestration/data/data_raw"
        PATH_DATA_PROCESSED = "/Users/mdurango/Proyect/Mlops-platzi/orchestration/data/data_processed"
        # reading JSON data
        data_tickets = self.read_json(
            path=PATH_DATA_RAW, file_name=f"{name_data_input}.json"
        )
        # data transformation
        data_tickets = self.data_transform(df=data_tickets)
        # data processing
        processed_column = self.text_preprocessing(
            data_tickets["complaint_what_happened"]
        )
        data_tickets["processed_text"] = processed_column
        # additional processing
        data_tickets["processed_text"] = data_tickets["processed_text"].str.replace(
            r"x+/", "", regex=True
        )
        data_tickets["processed_text"] = data_tickets["processed_text"].str.replace(
            "xxxx", ""
        )
        data_tickets = data_tickets.dropna(subset=["processed_text"])
        # Saving processed data
        self.save_processed_data(
            df=data_tickets,
            path=PATH_DATA_PROCESSED,
            file_name=f"{file_name}_{version}.csv",
        )
        self.logger.info(f"Data successfully saved to {PATH_DATA_PROCESSED}")


