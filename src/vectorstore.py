from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

def get_vectorstore_retriever(documents: list, persist_directory: str = "./chroma_db", embedding_model_name: str = "all-MiniLM-L6-v2", k: int = 3) -> VectorStoreRetriever:
    """
    Create and persist a Chroma vector store and return a retriever.
    
    Args:
        documents: List of document chunks to store.
        persist_directory: Directory to persist the vector store.
        embedding_model_name: Name of the HuggingFace embedding model.
        k: Number of documents to retrieve.
        
    Returns:
        A VectorStoreRetriever object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
