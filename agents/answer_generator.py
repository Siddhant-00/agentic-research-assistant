from langchain_core.prompts import ChatPromptTemplate


answer_prompt = ChatPromptTemplate.from_template(
    """
You are an expert AI research assistant.

Answer ONLY using the provided context.

If the answer is not present
say:

'I could not find enough information.'

Question:
{question}

Context:
{context}

Answer:
"""
)