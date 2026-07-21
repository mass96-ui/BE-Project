import ollama

def generate_answer(context, question):

    prompt = f"""
    You are an expert in biomechanics,
    EMG signal processing,
    muscle activation analysis,
    and rehabilitation engineering.

    Context:
    {context}

    Question:
    {question}

    Give a clear scientific answer based only on the context.
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]