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

print("SoleusAI Research Assistant Ready")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(
        question,
        k=3
    )

    print("\n========================")

    for i, doc in enumerate(docs):

        print(f"\nResult {i+1}\n")

        print(doc.page_content[:700])

    print("\n========================")