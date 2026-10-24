import ollama

from embeddings.runtime import DEFAULT_THREADS, validate_threads

LLM_MODEL = "qwen3"


def generate_answer(
    question,
    retrieved_chunks,
    num_threads=DEFAULT_THREADS,
    model=LLM_MODEL,
):
    """
    Generate a final answer from retrieved chunks.

    Args:
        question: User question.
        retrieved_chunks: List of retrieved text chunks.

    Returns:
        Generated answer.
    """

    validate_threads(num_threads)
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
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        options={
            "num_thread": num_threads,
            "num_ctx": 4096,
            "num_predict": 512,
        },
        keep_alive=0,
    )

    answer = response["message"]["content"]
    if not answer.strip():
        raise ValueError("Ollama returned an empty answer.")
    return answer