
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.db import engine, create_db_and_tables, User
from app.utils import run_preprocessing_fn
from sqlmodel import Session, select


app = FastAPI(title="FastAPI, Docker, and Traefik")
global label_mapping

label_mapping = {
    "0": "Bank Account Services",
    "1": "Credit Report or Prepaid Card",
    "2": "Mortgage/Loan"}


class ProcessTextRequestModel(BaseModel):
    sentences: list[str]

@app.post("/predict")
async def read_root(data: ProcessTextRequestModel):
    model = joblib.load("model.pkl")
    processed_data_vectorized = run_preprocessing_fn(data.sentences)
    X_dense = [sparse_matrix.toarray() for sparse_matrix in processed_data_vectorized]
    X_dense = np.vstack(X_dense) 
    preds = model.predict(X_dense) # a la mano de nuestro señor celestial, por favor señoooooooorrrrrrr 
    decoded_predictions = [label_mapping[str(pred)] for pred in preds]
    return {"predictions": decoded_predictions}

@app.post("/users/")
def create_user(user: User):
    with Session(engine) as session:
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

@app.get("/users/")
async def read_users():
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return users

@app.on_event("startup")
async def startup():
    create_db_and_tables()
