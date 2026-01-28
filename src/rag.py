from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(retriever, model_name: str = "llama3.2", temperature: float = 0):
    """
    Create a RAG chain using the given retriever and Ollama model.
    
    Args:
        retriever: VectorStoreRetriever object.
        model_name: Name of the Ollama model to use.
        temperature: Temperature for the model generation.
        
    Returns:
        A compiled RAG chain.
    """
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    Do not hallucinate or make up information apart from the context provided.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
