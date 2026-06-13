import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-key')
    # Use SQLite by default, swap manually if needed
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLITE_URL', 'sqlite:///neuroscreen.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False