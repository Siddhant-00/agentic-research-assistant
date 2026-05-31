from database.vector_db import get_retriever

retriever = get_retriever()

docs = retriever.invoke(
    "What are agent memory types?"
)

print(docs[0].metadata)