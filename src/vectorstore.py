import os
import sys

# =============================================
# CRITICAL: Set these BEFORE any other imports
# Fixes for Windows kernel crashes with torch/HuggingFace
# =============================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

def get_vectorstore_retriever(documents: list, persist_directory: str = "./faiss_index", embedding_model_name: str = "all-MiniLM-L6-v2", k: int = 3) -> VectorStoreRetriever:
    """
    Create and persist a FAISS vector store and return a retriever.
    
    Args:
        documents: List of document chunks to store.
        persist_directory: Directory to persist the vector store.
        embedding_model_name: Name of the HuggingFace embedding model.
        k: Number of documents to retrieve.
        
    Returns:
        A VectorStoreRetriever object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    
    # Check if index already exists
    index_path = os.path.join(persist_directory, "index.faiss")
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        # Persist the index
        os.makedirs(persist_directory, exist_ok=True)
        vectorstore.save_local(persist_directory)
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
