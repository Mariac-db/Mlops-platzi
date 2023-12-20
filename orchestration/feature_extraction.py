""" This module is used to extract features from the text data"""
import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore")


class FeatureExtraction:
    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(min_df=2, max_df=0.95)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def read_csv(self, path: str, file_name: str) -> pd.DataFrame:
        """Reads the CSV file and returns a DataFrame."""
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)
        return df

    def fit(self, df: pd.DataFrame) -> None:
        """Fits the TF-IDF vectorizer for a Document-Term Matrix."""
        # be sure not use NaN values
        self.df = df
        self.df = self.df.dropna(subset=["processed_text"])
        self.dtm = self.tfidf.fit_transform(self.df["processed_text"])
        len_vocab = len(self.tfidf.vocabulary_)
        self.logger.info(f"TF-IDF vectorizer fitted with {len_vocab}  unique words")

    def topic_modeling_nmf(self, n_components: int, num_words: int = 15) -> list:
        """Performs topic modeling using NMF and returns a list of topics.

        Args:
            n_components (int): Number of topics to extract.
            num_words (int, optional): Number of top words to display for each topic. Defaults to 10.

        Returns:
            list: List of strings, each string represents a topic consisting of top words.
        """
        self.nmf = NMF(n_components=n_components, random_state=123)
        self.nmf.fit(self.dtm)
        self.W = self.nmf.fit_transform(self.dtm)
        self.H = self.nmf.components_
        vocab = np.array(self.tfidf.get_feature_names_out())
        top_words = lambda t: [vocab[i] for i in np.argsort(t)[: -num_words - 1 : -1]]
        topic_words = [top_words(t) for t in self.H]
        topics = [" ".join(t) for t in topic_words]
        return topics

    def create_topics(self) -> pd.DataFrame:
        """This method is used to create a dataframe with the topics and the tickets.
        It is really import know that the number of topics returned by the NMF model will depend of the quality of the data
        Returns:
            pd.DataFrame: Dataframe with the topics and the tickets"""
        col_names = ["topic" + str(i) for i in range(self.nmf.n_components)]
        tickets_names = ["ticket_" + str(i) for i in range(len(self.df.processed_text))]
        df_doc_topics = pd.DataFrame(
            np.round(self.W, 2), columns=col_names, index=tickets_names
        )
        top_topics = np.argmax(self.df.values, axis=1)
        df_doc_topics["relevant_topics"] = top_topics
        print(df_doc_topics.head())
        self.df["relevant_topics"] = top_topics
        return self.df

    def topic_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """This method is used to map the topics with the tickets"""
        dict_mapping = {
            0: "Bank Account Services",
            1: "Credit Report or Prepaid Card",
            2: "Mortgage/Loan",
        }
        df["relevant_topics"] = df["relevant_topics"].map(dict_mapping)
        return dict_mapping  # df, self.dict_mapping

    def save_topic_mapping_to_json(self, dictionary: dict, path: str, file_name: str):
        """This method saves the dictionary to a JSON file"""
        file_path = os.path.join(path, file_name)
        with open(file_path, "w") as file:
            json.dump(dictionary, file, ensure_ascii=False)
        self.logger.info(f"Dictionary successfully saved to {file_path}")

    def save_df_to_csv(self, df: pd.DataFrame, path: str, file_name: str):
        """This method saves the dataframe to a CSV file"""
        file_path = os.path.join(path, file_name)
        df.to_csv(file_path, index=False)
        self.logger.info(f"Dataframe successfully saved to {file_path}")

    def run(self, data_path_processed: str, data_version: int):
        df_tickets = self.read_csv(
            path=data_path_processed,
            file_name=f"tickets_classification_eng_{data_version}.csv",
        )
        self.fit(df_tickets)
        extracted_topics = self.topic_modeling_nmf(n_components=4)
        for idx, topic in enumerate(extracted_topics):
            print(f"Topic {idx}: {topic}")
        df_tickets = self.create_topics()
        topic_mapping = self.topic_mapping(df_tickets)
        self.save_topic_mapping_to_json(
            dictionary=topic_mapping,
            path=data_path_processed,
            file_name=f"topic_mapping_{data_version}.json",
        )
        self.save_df_to_csv(
            df_tickets, data_path_processed, f"tickets_inputs_eng_{data_version}.csv"
        )




