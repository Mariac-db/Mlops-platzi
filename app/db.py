from .config import settings

from sqlmodel import Field, Session, SQLModel, create_engine, select


class PredictionsTickets(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    client_name: str 
    prediction: str

# connect_args = {"check_same_thread": False}
engine = create_engine(settings.db_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
