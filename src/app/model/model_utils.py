import os
import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import warnings

warnings.filterwarnings("ignore")


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
    return X_train.tolist(), X_val.tolist(), X_test.tolist(), y_train, y_val, y_test


def tokenize_texts(tokenizer, texts:str , max_length=415):
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
    """This functions trains the model
    Args:
        model (BertForSequenceClassification): model
        optimizer (torch.optim): optimizer
        loss_fn (torch.nn): loss function
        train_loader (DataLoader): train dataloader
        val_loader (DataLoader): validation dataloader
        num_epochs (int): number of epochs
        device (torch.device): device
    Returns:
        None
    """
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch[:-1]}
            labels = batch[-1].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        val_accuracies = []
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch[:-1]}
            labels = batch[-1].to(device)

            with torch.no_grad():
                outputs = model(**inputs)
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

def test_model(model, test_loader, loss_fn, device):
    """Esta función evalúa el modelo en el conjunto de prueba"""
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch[:-1]}
            labels = batch[-1].to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return avg_loss, all_labels, all_predictions


def run_training(num_labels= 3, num_epochs=3, test_size=0.2, val_size=0.2, random_state=42):

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = prepare_data(
        test_size=test_size, val_size=val_size, random_state=random_state
        
    )
    print(type(train_texts))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings = tokenize_texts(tokenizer, val_texts)
    test_encodings = tokenize_texts(tokenizer, test_texts)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device)
    test_loss, test_labels, test_predictions = test_model(model, test_loader, loss_fn, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy_score(test_labels, test_predictions):.4f}")





run_training(num_labels= 3, num_epochs=3, test_size=0.2, val_size=0.2, random_state=42)






