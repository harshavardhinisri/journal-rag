import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv() 

# --- Config ---
PROJECT_ROOT = "/Users/harsha/projects/journal-rag"
PERSIST_DIR = os.path.join(PROJECT_ROOT, "embeddings")
RETRIEVAL_K = 5

SYSTEM_PROMPT = """You are an AI assistant that answers questions based on a user's journal entries.

Context:
{context}

Answer the question using only the context above. If the context doesn't contain the answer, say you don't know.
"""

# --- Load vector DB ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# --- Load local LLM ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# --- Helper functions ---
def fetch_context(question: str):
    """Retrieve relevant documents for a question."""
    print("Retrieved docs:", len(retriever.invoke(question)))
    return retriever.invoke(question)

def answer_question(question: str):
    """Answer the question using RAG logic."""
    docs = fetch_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    print(context)

    prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    return response.content, docs

# --- Example usage ---
if __name__ == "__main__":
    query = "What workout did I do in the first week of January?"
    answer, source_docs = answer_question(query)
    
    print("\n=== Answer ===\n")
    print(answer)
    
    print("\n=== Sources ===\n")
    for doc in source_docs:
        print(f"Date: {doc.metadata['date']}\nCategory: {doc.metadata['category']}\nText: {doc.page_content}\n")