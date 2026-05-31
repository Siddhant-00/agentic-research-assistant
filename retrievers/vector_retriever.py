from database.vector_db import get_retriever


def retrieve_from_vectorstore(question):

    retriever = get_retriever()

    docs = retriever.invoke(question)

    return docs