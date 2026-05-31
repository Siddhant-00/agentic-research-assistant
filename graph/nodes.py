from agents.query_rewriter import rewrite_prompt
from agents.answer_generator import answer_prompt
from retrievers.vector_retriever import (
    retrieve_from_vectorstore
)
from retrievers.reranker import (
    rerank_documents
)
from langchain_groq import ChatGroq

from agents.verifier import verifier_prompt
import os

from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def rewrite_query(state):
    """
    Rewrite user query
    """

    question = state["question"]

    chat_history = state.get(
    "chat_history",
    []
    )

    chain = rewrite_prompt | llm

    rewritten_question = chain.invoke(
        {"question": question, 
        "chat_history": chat_history}
    ).content

    return {
        "rewritten_question": rewritten_question
    }

from agents.router import RouteQuery

from retrievers.vector_retriever import (
    retrieve_from_vectorstore
)

from retrievers.wiki_retriever import (
    retrieve_from_wikipedia
)

from retrievers.arxiv_retriever import (
    retrieve_from_arxiv
)

from langchain_core.prompts import ChatPromptTemplate
# router llm
router_llm = llm.with_structured_output(
    RouteQuery
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           """
You are an intelligent routing agent.

Route queries as follows:

vectorstore:
- agent architectures
- prompt engineering
- adversarial attacks
- content already stored in our knowledge base

arxiv:
- research papers
- scientific publications
- machine learning papers
- deep learning papers
- LLM papers
- AI research

wiki_search:
- people
- places
- companies
- movies
- historical events
- general knowledge

Return only the datasource.
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)

router_chain = router_prompt | router_llm

# routenode
def route_question(state):

    question = state["rewritten_question"]

    source = router_chain.invoke(
        {"question": question}
    )

    return {
        "source": source.datasource
    }

# retrieval node
def retrieve_documents(state):

    source = state["source"]

    question = state["rewritten_question"]

    if source == "vectorstore":

        docs = retrieve_from_vectorstore(
            question
        )

    elif source == "wiki_search":

        docs = retrieve_from_wikipedia(
            question
        )

    else:

        docs = retrieve_from_arxiv(
            question
        )

    return {
    "documents": docs,
    "source": source
}
# reranking node
def rerank_retrieved_documents(state):

    question = state["question"]

    docs = state["documents"]

    reranked_docs = rerank_documents(
        question,
        docs,
        top_k=3
    )

    return {
        "reranked_documents": reranked_docs
    }
# answer node
def generate_answer(state):

    question = state["question"]

    docs = state["reranked_documents"]

    context = "\n\n".join(
        [
            doc.page_content[:3000]
            for doc in docs
        ]
    )

    chain = answer_prompt | llm

    answer = chain.invoke(
        {
            "question": question,
            "context": context
        }
    ).content

    # Collect source metadata
    sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown Source"
        )

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": list(set(sources))
    }

def verify_answer(state):

    question = state["question"]

    docs = state["reranked_documents"]

    answer = state["answer"]

    documents_text = "\n\n".join(
        [
            doc.page_content[:2000]
            for doc in docs
        ]
    )

    chain = verifier_prompt | llm

    verified_answer = chain.invoke(
        {
            "question": question,
            "documents": documents_text,
            "answer": answer
        }
    ).content

    return {
        "verified_answer": verified_answer,
        "sources": state["sources"]
    }