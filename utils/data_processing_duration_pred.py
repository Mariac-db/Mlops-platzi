"""This module contains ProcessingData class for data trips"""

import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingData:
    """
    A class for processing and pre-processing data.

    Attributes:
        None

    Methods:
        dump_pickle(obj, filename):
            Dump an object to a pickle file.

        read_dataframe(filename):
            Read and preprocess a DataFrame from a file.

        preprocess(df, dv, fit_dv=False):
            Preprocess the DataFrame and return transformed data.

        run(dest_path="data_model_input", dataset="green", raw_data_path="data"):
            Process the dataset and save preprocessed data and artifacts.
    """

    def __init__(self):

        pass

    def dump_pickle(self, obj: object, filename: str):
        """
        Dump an object to a pickle file.

        Args:
            obj (object): The object to dump.
            filename (str): The name of the output pickle file.
        Returns:
            None
        """
        with open(filename, "wb") as f_out:
            return pickle.dump(obj, f_out)

    def read_dataframe(self, filename: str):
        """
        Read and preprocess a DataFrame from a file.

        Args:
            filename (str): The name of the input data file.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = pd.read_parquet(filename)
        df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        categorical = ["PULocationID", "DOLocationID"]
        df[categorical] = df[categorical].astype(str)
        return df

    def preprocess(self, df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
        """
        Preprocess the DataFrame and return transformed data.

        Args:
            df (pd.DataFrame): The input DataFrame.
            dv (DictVectorizer): The DictVectorizer for categorical feature encoding.
            fit_dv (bool, optional): Whether to fit the DictVectorizer. Defaults to False.

        Returns:
            tuple: A tuple containing transformed data and the fitted DictVectorizer.
        """
        df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
        categorical = ["PU_DO"]
        numerical = ["trip_distance"]
        dicts = df[categorical + numerical].to_dict(orient="records")
        if fit_dv:
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
        return X, dv

    def run(
        self,
        dest_path="tracking/data/data_processed",
        dataset="green",
        raw_data_path="tracking/data/data_raw",
    ):
        """
        Process the dataset and save preprocessed data and artifacts.

        Args:
            dest_path (str, optional): The destination path for saving preprocessed data. 
            Defaults to "data/data_processed"
            dataset (str, optional): The dataset to process. Defaults to "green".
            raw_data_path (str, optional): The path to raw data files. Defaults to "data".

        Returns:
            None
        """

        df_train = self.read_dataframe(
            os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet")
        )
        df_valid = self.read_dataframe(
            os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet")
        )
        df_test = self.read_dataframe(
            os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
        )
        print(df_test.head(2)) #print como pruebita para saber que todo está good
        
        target = "duration"
        y_train = df_train[target].values
        y_valid = df_valid[target].values
        y_test = df_test[target].values
        dv = DictVectorizer()
        X_train, dv = self.preprocess(df_train, dv, fit_dv=True)
        X_valid, _ = self.preprocess(df_valid, dv, fit_dv=False)
        X_test, _ = self.preprocess(df_test, dv, fit_dv=False)
        os.makedirs(dest_path, exist_ok=True)

        self.dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
        self.dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
        self.dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))
        self.dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
        logger.info(f"Data successfully processed and saved in {dest_path}")


if __name__ == "__main__":
    data_processor = ProcessingData()
    data_processor.run()
