# Standard library imports
import os
import warnings

# Third party imports
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support, 
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from prefect import flow, task

# Local application imports
from config import *
from utils import * 
from feature_extraction import FeatureExtraction
from text_processing import TextProcessing



warnings.filterwarnings("ignore")


@task(retries=3, retry_delay_seconds=2,
      name="Text processing task", 
      tags=["pos_tag"])
def text_processing_task(language: str, file_name: str, version: int):
    """This task is used to run the text processing process
    Args:
        language (str): language of the text
        file_name (str): file name of the data
        version (int): version of the data
    Returns:
        None"""
    text_processing_processor = TextProcessing(language=language)
    text_processing_processor.run(file_name=file_name, version=version)

@task(retries=3, retry_delay_seconds=2,
      name="Feature extraction task", 
      tags=["feature_extraction", "topic_modeling"])
def feature_extraction_task(data_path_processed: str, 
                            data_version: int):
    """This task is used to run the feature extraction process
    Args:
        data_path_processed (str): path where the data is stored
        data_version (int): version of the data
    Returns:
        None"""
    feature_extraction_processor = FeatureExtraction()
    feature_extraction_processor.run(data_path_processed=data_path_processed, 
                                     data_version = VERSION)

@task(retries=3, retry_delay_seconds=2,
      name="Data transformation task", 
      tags=["data_transform", "split_data", "train_test_split"])    
def data_transformation_task_and_split(data_input_path: str, file_name: str, version: int):
    """This function transform the data into X and y
    Args:
      df (pd.DataFrame): dataframe with the data
    Returns:
      X (pd.Series): series with the text
      y (pd.Series): series with the labels"""
    
    # read data
    df = pd.read_csv(os.path.join(data_input_path, f"{file_name}{version}.csv"))
    X = df["processed_text"]
    y = df["relevant_topics"]
    # feature extraction from text input
    count_vectorizer = CountVectorizer()
    X_vectorized = count_vectorizer.fit_transform(X)
    # transform labels into idx for model input
    y = decode_labels_into_idx(labels=y, idx2label=label2idx)
    # transform into tfidf and split data
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_vectorized)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.3, random_state=42
    )
    print("Data transformation and split task successfully completed")
    return X_train, X_test, y_train, y_test, count_vectorizer

@task(
    retries=3,
    retry_delay_seconds=2,
    name="Train best model",
    tags=["train", "best_model", "LogisticRegressionClassifier"],
)
def training_best_model(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    params: dict,
    model_name: str,
):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("developer", DEVELOPER_NAME)
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_train_pred_proba = model.predict_proba(X_train)
        y_test_pred_proba = model.predict_proba(X_test)

        roc_auc_score_train = round(
            roc_auc_score(
                y_train, y_train_pred_proba, average="weighted", multi_class="ovr"
            ),
            2,
        )
        roc_auc_score_test = round(
            roc_auc_score(
                y_test, y_test_pred_proba, average="weighted", multi_class="ovr"
            ),
            2,
        )

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        precision_train, recall_train, fscore_train, support_train = precision_recall_fscore_support(
            y_train, y_train_pred, average="weighted"
        )
        precision_test, recall_test, fscore_test, support_test = precision_recall_fscore_support(
            y_test, y_test_pred, average="weighted"
        )

        mlflow.log_metrics(
            {
                "roc_auc_train": roc_auc_score_train,
                "roc_auc_test": roc_auc_score_test,
                "precision_train": precision_train,
                "precision_test": precision_test,
            }
        )

        mlflow.sklearn.log_model(model, f"model_{MODEL_NAME}")

        # save model
        save_pickle(model, "model_lr")

        metric_data = [
            roc_auc_score_train,
            roc_auc_score_test,
            round(precision_train, 2),
            round(precision_test, 2),
            round(recall_train, 2),
            round(recall_test, 2),
            round(fscore_train, 2),
            round(fscore_test, 2),
        ]

        print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")

        model_report_train = classification_report(y_train, y_train_pred)
        model_report_test = classification_report(y_test, y_test_pred)

        print("Classification Report for Train:\n", model_report_train)
        print("Classification Report for Test:\n", model_report_test)

        return metric_data


@flow
def main_flow():
    text_processing_task(language = LANGUAGE, file_name = FILE_NAME_DATA_INPUT, version = VERSION)
    feature_extraction_task(data_path_processed = DATA_PATH_PROCESSED,
                            data_version = VERSION)
    X_train, X_test, y_train, y_test, count_vectorizer = data_transformation_task_and_split(
        data_input_path=DATA_PATH_PROCESSED,
        file_name="tickets_inputs_eng_",
        version=VERSION
    )
    save_pickle((X_train, y_train), "train")
    save_pickle((X_test, y_test),  "test")
    save_pickle(count_vectorizer, "count_vectorizer")
    print("Data transformation and split task successfully completed and stored in pickle files")

    metrics_classification = training_best_model(X_train = X_train, 
                                                y_train = y_train,
                                                X_test = X_test,
                                                y_test = y_test,
                                                params = PARAMETERS_MODEL,
                                                model_name = MODEL_NAME)
    print(metrics_classification)


main_flow()

