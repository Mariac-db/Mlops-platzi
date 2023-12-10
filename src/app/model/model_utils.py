"""This module contains the functions to train the model"""

import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import warnings
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay


warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("tickets_classification_bert")


def load_data(
    file_name: str, path="src/app/model/data/data_preprocessed"
) -> pd.DataFrame:
    """
    Read csv file
    :param file_name: file name
    :param path: path to the file
    :return: pandas dataframe
    """
    df = pd.read_csv(os.path.join(path, file_name))
    X = df["processed_text"]
    y = df["relevant_topics"]
    return X, y


def read_idx2label(json_path: str) -> dict:
    """This function reads the json file and returns a dictionary
    Args:
        json_path (str): path to the json file
    Returns:
        idx2label (dict): dictionary with the mapping
    """
    with open(json_path) as f:
        idx2label = json.load(f)
    return idx2label


def decode_labels_into_idx(labels: pd.Series, idx2label: dict) -> pd.Series:
    """This function decodes the labels into idx
    Args:
        labels (pd.Series): series with the labels
        idx2label (dict): dictionary with the mapping
    Returns:
        labels (pd.Series): series with the labels decoded
    """
    return labels.map(idx2label)


def prepare_data(test_size=0.2, val_size=0.2, random_state=42):
    """This function prepares the data
    Args:
        test_size (float): test size
        val_size (float): validation size
        random_state (int): random state
    Returns:
        X_train (list): list of texts for training
        X_val (list): list of texts for validation
        X_test (list): list of texts for testing
        y_train (list): list of labels for training
        y_val (list): list of labels for validation
        y_test (list): list of labels for testing
    """
    global idx2label

    idx2label = read_idx2label(
        json_path="src/app/model/data/data_preprocessed/topic_mapping_1.json"
    )
    label2idx = {value: key for key, value in idx2label.items()}
    X, y = load_data("tickets_inputs_eng_1.csv")
    y = decode_labels_into_idx(labels=y, idx2label=label2idx)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    X_train = X_train.tolist()
    X_val = X_val.tolist()
    X_test = X_test.tolist()

    y_train = y_train.astype("int64")
    y_val = y_val.astype("int64")
    y_test = y_test.astype("int64")

    return X_train, X_val, X_test, y_train, y_val, y_test


def tokenize_texts(tokenizer, texts: str, max_length=415):
    """This function tokenizes the texts
    Args:
         tokenizer (BertTokenizer): tokenizer
         texts (list): list of texts
        max_length (int): maximum length of the text"""
    return tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )


def train_model(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device
):
    """This functions trains the model."""
    for epoch in range(num_epochs):
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        val_accuracies = []

        for batch in tqdm(
            val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"
        ):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = loss_fn(outputs.logits, labels)
                val_losses.append(val_loss.item())

                _, predicted = torch.max(outputs.logits, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / labels.size(0)
                val_accuracies.append(accuracy)

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}"
        )

    return model


def test_model(model, test_loader, loss_fn, device):
    """This function evaluates the model on the test set and returns the metrics"""
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    
    return avg_loss, accuracy, precision, recall, all_labels, all_predictions

def decode_idx_into_labels(labels, idx2label):
    """Decode labels into their original values using the idx2label dictionary
        Args: 
            - labels (int[list]): labels can be predctions or true labels
            - idx2label (dict): with id to labels mapping"""
    decoded_labels = [idx2label[str(label_idx)] for label_idx in labels]
    return decoded_labels

def run_training(
    num_labels=3,
    num_epochs=1,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    run_name=None,
):
    (
        train_texts,
        val_texts,
        test_texts,
        train_labels,
        val_labels,
        test_labels,
    ) = prepare_data(test_size=test_size, val_size=val_size, random_state=random_state)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings = tokenize_texts(tokenizer, val_texts)
    test_encodings = tokenize_texts(tokenizer, test_texts)

    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        torch.tensor(train_labels),
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        torch.tensor(val_labels),
    )
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(test_labels),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "num_labels": num_labels,
                "num_epochs": num_epochs,
                "test_size": test_size,
                "val_size": val_size,
                "random_state": random_state,
            }
        )

        model = train_model(
            model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device
        )
        mlflow.pytorch.log_model(model, "model")

        test_loss, test_accuracy, test_precision, test_recall, test_labels, test_predictions = test_model(
            model, test_loader, loss_fn, device
        )

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall, 
            }
        )
        mlflow.set_tag("mlflow.runName", run_name)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        test_labels_decoded = decode_idx_into_labels(test_labels, idx2label)
        test_predictions_decoded = decode_idx_into_labels(test_labels, idx2label)

        fig, ax = plt.subplots(figsize=(12, 8))
        cm = confusion_matrix(test_labels_decoded, test_predictions_decoded)
        cmp = ConfusionMatrixDisplay(cm, display_labels= list(idx2label.values()))
        cmp.plot(ax=ax)
        plt.xticks(rotation=80)
        plt.show()

        # Plotear matriz de confusión
        cm = confusion_matrix(test_labels, test_predictions)
        plot_confusion_matrix(cm, label_names=["Class 0", "Class 1", "Class 2"])

def plot_confusion_matrix(confusion_matrix, label_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(label_names)), yticks=range(len(label_names)),
           xticklabels=label_names, yticklabels=label_names,
           title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.show()

run_training(
    num_labels=3,
    num_epochs=5,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    run_name="BertUncased",
)

# mlflow ui --backend-store-uri sqlite:///backend.db