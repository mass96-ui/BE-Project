import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Starting...")

documents = []

pdf_folder = "src/rag/papers"

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        print(f"Loading {file}")

        path = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(path)

        docs = loader.load()

        documents.extend(docs)

print(f"Pages Loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embedding
)

db.save_local("vectorstore")

print("Knowledge Base Created Successfully")