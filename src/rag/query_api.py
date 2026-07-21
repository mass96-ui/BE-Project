from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.rag.llm_rag import generate_answer

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)

def search_papers(question):

    docs = db.similarity_search(
        question,
        k=3
    )

    context = ""

    for doc in docs:
        context += doc.page_content
        context += "\n\n"

    answer = generate_answer(
        context,
        question
    )

    return answer