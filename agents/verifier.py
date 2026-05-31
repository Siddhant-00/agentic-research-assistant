from langchain_core.prompts import ChatPromptTemplate

verifier_prompt = ChatPromptTemplate.from_template(
    """
You are a factual verification expert.

Your job is to verify whether the answer is supported by the provided documents.

Question:
{question}

Documents:
{documents}

Answer:
{answer}

Instructions:

1. Verify factual correctness.
2. Remove unsupported claims.
3. Correct mistakes.
4. Keep the answer concise.
5. Return ONLY the corrected answer.

Verified Answer:
"""
)