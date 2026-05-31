from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "faiss_index"


def load_vectorstore():

    if not Path(DB_PATH).exists():

        raise Exception(
            """
FAISS index not found.

Run:

python database/ingest.py
"""
        )

    vectorstore = FAISS.load_local(
        DB_PATH,
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def get_retriever():

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )