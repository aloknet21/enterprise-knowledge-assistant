import ollama


LLM_MODEL = "qwen3"


def generate_answer(
    question,
    retrieved_chunks
):
    """
    Generate a final answer from retrieved chunks.

    Args:
        question: User question.
        retrieved_chunks: List of retrieved text chunks.

    Returns:
        Generated answer.
    """

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an Enterprise Knowledge Assistant.

Answer the user's question only using the information
provided in the context below.

If the answer is not present in the context,
say:

"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Provide:
1. A clear answer
2. A concise summary
3. Bullet points where appropriate
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]