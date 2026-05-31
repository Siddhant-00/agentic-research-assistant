from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):

    question: str

    chat_history: List

    rewritten_question: str

    source: str

    documents: List

    reranked_documents: List

    answer: str

    verified_answer: str

    sources: List[str]