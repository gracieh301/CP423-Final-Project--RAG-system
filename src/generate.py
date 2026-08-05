import ollama

# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "llama3.2"

TEMPERATURE = 0

MAX_TOKENS = 300

SEED = 42  # fixed seed so Ollama generations are reproducible run to run


# -------------------------
# Build prompt
# -------------------------

def build_prompt(question, retrieved_chunks):
    """
    Build a RAG prompt from the retrieved chunks.
    """

    context = ""

    for chunk in retrieved_chunks:

        context += (
            f"[{chunk['chunk_id']}]\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.

Answer ONLY using the information contained in the provided context documents.

Rules:

1. Do NOT use outside knowledge.
2. If the answer is not completely supported by the context, reply exactly:

I don't know.

3. Cite every factual statement using the corresponding chunk ID in square brackets.
Example:
Patrick Star is SpongeBob's best friend. [CHAR4_001]

4. Do not invent citations.

Context
-------
{context}

Question
--------
{question}

Answer
"""

    return prompt.strip()


# -------------------------
# Generate answer (RAG: retrieval + LLM)
# -------------------------

def generate_answer(question, retrieved_chunks):

    prompt = build_prompt(question, retrieved_chunks)

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options={

            "temperature": TEMPERATURE,

            "num_predict": MAX_TOKENS,

            "seed": SEED

        }

    )

    return response["message"]["content"]


# -------------------------
# Generate answer (LLM ONLY -- no retrieval, no context)
# -------------------------

def generate_plain_answer(question):
    """
    Baseline: ask the LLM the question directly, with no retrieved
    context at all. Used to compare RAG vs. no-RAG performance.
    """

    prompt = f"""
You are a helpful assistant. Answer the following question as best
you can using your own knowledge.

Question
--------
{question}

Answer
"""

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt.strip()
            }
        ],

        options={

            "temperature": TEMPERATURE,

            "num_predict": MAX_TOKENS,

            "seed": SEED

        }

    )

    return response["message"]["content"]


# -------------------------
# Example
# -------------------------

if __name__ == "__main__":

    question = "Who is SpongeBob's best friend?"

    retrieved_chunks = [

        {

            "chunk_id": "CHAR4_001",

            "text": "Patrick Star is SpongeBob's best friend. Patrick lives under a rock."

        },

        {

            "chunk_id": "CHAR5_002",

            "text": "SpongeBob and Patrick spend much of their time together."

        }

    ]

    answer = generate_answer(

        question,

        retrieved_chunks

    )

    print("\nGenerated answer (RAG):\n")

    print(answer)

    plain_answer = generate_plain_answer(question)

    print("\nGenerated answer (LLM only):\n")

    print(plain_answer)