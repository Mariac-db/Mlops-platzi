import os

from pydantic import BaseSettings, Field

# fields validation
# base settings as class for defining configuration and field for defining fields with theirs validations
class Settings(BaseSettings):
    """
    Base Settings Class for Configuration Definition
    
    This class is used to define configuration settings for the application.
    """
    
    db_url: str = Field(..., env='DATABASE_URL')


settings = Settings()
