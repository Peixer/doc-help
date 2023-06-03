import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)
