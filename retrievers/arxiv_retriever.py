from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.documents import Document

arxiv = ArxivAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=3000
)


def retrieve_from_arxiv(question):

    result = arxiv.run(question)

    return [
        Document(
            page_content=result,
            metadata={
                "source": "Arxiv"
            }
        )
    ]