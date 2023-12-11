
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from app.db import engine, create_db_and_tables, User
from app.utils import run_preprocessing_fn
from sqlmodel import Session, select


app = FastAPI(title="FastAPI, Docker, and Traefik")


class ProcessTextRequestModel(BaseModel):
    sentences: list[str]

@app.post("/predict")
async def read_root(data: ProcessTextRequestModel):
    model = joblib.load("model.pkl")
    processed_data_vectorized = run_preprocessing_fn(data.sentences)
    preds = model.predict(processed_data_vectorized) # a la mano de nuestro señor celestial, por favor señoooooooorrrrrrr 
    return {"predictions": preds}
#TODO:  hacer decodings de las labels 

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

 # uvicorn app.main:app