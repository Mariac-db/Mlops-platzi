"""This config module contains the configuration for the pipeline with prefect"""

# path with data processed
DATA_PATH_PROCESSED = "/Users/mdurango/Proyect/Mlops-platzi/orchestration/data/data_processed"
# version of the data
VERSION = 2
# language for the input parameter for the text processing class
LANGUAGE = "english"
# file name data input for the text processing class
FILE_NAME_DATA_INPUT = "tickets_classification_eng"
# parameters for the logistic regression model based on the model training with mlflow
PARAMETERS_MODEL = {
    "C": 1.0,
    "class_weight": None,
    "l1_ratio": None,
    "max_iter": 100,
    "penalty": "l2",
    "random_state": 40,
    "solver": "liblinear",
    "tol": 0.0001,
}

idx2label = {"0": "Bank Account Services", "1": "Credit Report or Prepaid Card", "2": "Mortgage/Loan"}
label2idx = {v: k for k, v in idx2label.items()}
# tags for mlflow tracking
DEVELOPER_NAME = "Maria"
MODEL_NAME = "LogisticRegression"