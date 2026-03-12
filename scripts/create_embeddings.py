# scripts/create_embeddings.py

import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Absolute paths (to match your create_documents script)
project_root = "/Users/harsha/projects/journal-rag"
docs_path = os.path.join(project_root, "data/processed/journal_docs.json")
persist_dir = os.path.join(project_root, "embeddings")

# Load processed documents
with open(docs_path, "r") as f:
    docs = json.load(f)

# Extract text and metadata
texts = [doc["text"] for doc in docs]
metadatas = [doc["metadata"] for doc in docs]

# Initialize FREE local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Create vector database
vectordb = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory=persist_dir
)

# Persist embeddings
vectordb.persist()

print(f"Vector DB created with {len(texts)} documents and stored at {persist_dir}")