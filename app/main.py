import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.db import engine, create_db_and_tables, PredictionsTickets
from app.utils import preprocessing_fn
from sqlmodel import Session, select
from enum import Enum

app = FastAPI(title="FastAPI, Docker, and Traefik")
global label_mapping

label_mapping = {
    "0": "Bank Account Services",
    "1": "Credit Report or Prepaid Card",
    "2": "Mortgage/Loan"}


class Sentence(BaseModel):
    client_name: str
    text: str 

class ProcessTextRequestModel(BaseModel):
    sentences: list[Sentence]


@app.post("/predict")
async def read_root(data: ProcessTextRequestModel):

    session = Session(engine)
    
    model = joblib.load("model.pkl")

    preds_list = []

    for sentence in data.sentences: 
        processed_data_vectorized = preprocessing_fn(sentence.text)
        X_dense = [sparse_matrix.toarray() for sparse_matrix in processed_data_vectorized]
        X_dense = np.vstack(X_dense) 

        preds = model.predict(X_dense)
        decoded_predictions = label_mapping[str(preds[0])]

        
        prediction_ticket = PredictionsTickets(
            client_name=sentence.client_name,
            prediction=decoded_predictions
        )
        
        print(prediction_ticket)

        preds_list.append({
            "client_name": sentence.client_name,
            "prediction": decoded_predictions
        })
        
        session.add(prediction_ticket)

    session.commit()
    session.close()

    return {"predictions": preds_list}



@app.on_event("startup")
async def startup():
    create_db_and_tables()
