from langchain_core.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_template(
    """
You are a query rewriting expert.

Given:

Conversation History:
{chat_history}

Current Question:
{question}

Rewrite the current question so it becomes fully self-contained.

Do not answer the question.

Return only the rewritten query.
"""
)