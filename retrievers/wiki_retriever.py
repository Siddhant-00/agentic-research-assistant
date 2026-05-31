from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.documents import Document

wiki = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=3000
)


def retrieve_from_wikipedia(question):

    result = wiki.run(question)

    return [
        Document(
            page_content=result,
            metadata={
                "source": "Wikipedia"
            }
        )
    ]