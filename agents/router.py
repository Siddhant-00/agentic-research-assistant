from typing import Literal
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """
    Route question to correct datasource
    """

    datasource: Literal[
        "vectorstore",
        "wiki_search",
        "arxiv"
    ] = Field(
        ...,
        description="""
        Route to:

        vectorstore:
        agent systems,
        prompt engineering,
        adversarial attacks

        wiki_search:
        general knowledge

        arxiv:
        research papers,
        AI papers,
        ML papers,
        DL papers,
        scientific papers
        """
    )