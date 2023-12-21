from .config import settings

from sqlmodel import Field, Session, SQLModel, create_engine, select


class PredictionsTickets(SQLModel, table=True):
    """
    Table Model for Prediction Tickets.

    This class represents a table in the database to store
    predictions associated with client names.
    """
    id: int | None = Field(default=None, primary_key=True)
    client_name: str 
    prediction: str

# connect_args = {"check_same_thread": False}
engine = create_engine(settings.db_url, echo=True)


def create_db_and_tables():
    """
    Create tables in the database.

    This function uses the database engine to create all tables
    defined by the models (in this case, only the PredictionsTickets table)
    in the database.
    """
    SQLModel.metadata.create_all(engine)
