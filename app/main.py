from fastapi import FastAPI

app = FastAPI(title="FastAPI, Docker, and Traefik")


@app.get("/")
def read_root():
    return {"hello": "world"}

 # uvicorn app.main:app