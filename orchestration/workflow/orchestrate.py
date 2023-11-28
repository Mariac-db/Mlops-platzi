""" This module contains a pipeline orchestartion for the duration prediction model with prefect
to run the pipeline wou should run the following command in a terminal:
 prefect server start
  then in another terminal you should be inside orchestration/workflow folder and
   run the script with python orchestrate.py """

import pathlib
import pickle
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
import xgboost as xgb
from prefect import flow, task


@task(
    name="Read Data Task",
    description="Reads data from a given path into a DataFrame, performs data cleaning, and returns the shape of the DataFrame.",
    tags=["data-processing", "data-cleaning", "pandas"],
    retries=3,
    retry_delay_seconds=2,
)
def read_data(data_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(data_path)
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


@task(
    name="Add Features Task",
    description="Adds features to the model using categorical and numerical data, returns transformed data and DictVectorizer.",
    tags=["feature-engineering", "model-preprocessing", "scikit-learn"],
)
def add_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Add features to the model"""
    print(df_train.head(2))
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]
    df_test["PU_DO"] = df_test["PULocationID"] + "_" + df_test["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]
    dv = DictVectorizer()

    # training dataset
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train["duration"].values

    # validation dataset
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_val = df_val["duration"].values

    # test dataset
    test_dicts = df_test[categorical + numerical].to_dict(orient="records")
    X_test = dv.transform(test_dicts)
    y_test = df_test["duration"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


@task(
    name="Train Best Model Task",
    description="Trains a model with best hyperparameters, logs metrics, and saves the model and preprocessor artifacts using MLflow.",
    tags=["model-training", "hyperparameter-tuning", "xgboost", "mlflow"],
    log_prints=True,
)
def train_best_model(
    X_train: scipy.sparse.csr_matrix,
    X_test: scipy.sparse.csr_matrix,
    X_val: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("model", "xgboost")
        mlflow.set_tag("developer", "Maria")

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        test = xgb.DMatrix(X_test, label=y_test)
        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=50,
            evals=[(valid, "validation")],
            early_stopping_rounds=5,
        )

        y_pred_val = booster.predict(valid)
        y_pred_test = booster.predict(test)
        rmse_valid = mean_squared_error(y_val, y_pred_val, squared=False)
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        mlflow.log_metric("rmse_valid", rmse_valid)
        mlflow.log_metric("rmse_test", rmse_test)
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
    return None


@flow(
    name="Workflow for xgboost traini model",
    log_prints=True,
    version=1,
    description="This is a workflow for NYC Taxi Data",
)
def main_flow(
    train_path: str = "../data/data_raw/green_tripdata_2023-01.parquet",
    val_path: str = "../data/data_raw/green_tripdata_2023-02.parquet",
    test_path: str = "../data/data_raw/green_tripdata_2023-03.parquet",
) -> None:
    """The main training pipeline for xgboost model with nyc taxi data"""

    # MLflow settings - local with mlflow.db -> mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-prefect")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)
    df_test = read_data(test_path)

    # Transform
    X_train, X_val, X_test, y_train, y_val, y_test, dv = add_features(
        df_train, df_val, df_test
    )

    # Train
    train_best_model(X_train, X_test, X_val, y_train, y_test, y_val, dv)


if __name__ == "__main__":
    main_flow()
