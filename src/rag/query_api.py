from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

    response = ""

    for doc in docs:
        response += doc.page_content
        response += "\n\n"

    return response