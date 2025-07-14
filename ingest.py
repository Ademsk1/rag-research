"""
This script:

Reads PDFs

Splits into chunks

Embeds with OpenAI

Stores in ChromaDB
"""

# ingest.py

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

def ingest():
    print("Loading documents...")
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)

    db.persist()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest()
