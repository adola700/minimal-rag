# Minimal RAG Solution

A simple Retrieval-Augmented Generation (RAG) system running locally.

## Components

| Component | Technology | Description |
|-----------|------------|-------------|
| Data Source | Wikipedia | Public knowledge base for demo |
| Text Splitter | LangChain | Breaks documents into chunks |
| Embeddings | all-MiniLM-L6-v2 | Converts text to vectors |
| Vector Store | FAISS | Fast similarity search |
| LLM | Ollama (Llama 3.2) | Local language model |

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (Windows)
Download from https://ollama.com/download, then:
```bash
ollama pull llama3.2:1b
# If not in PATH, try:
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.2:1b
```

### 3. Run the notebook
Open `rag_solution.ipynb` and run all cells.

## Project Structure

```
minimal-rag/
├── rag_solution.ipynb    # Main notebook
├── src/
│   ├── splitter.py       # Text chunking
│   ├── vectorstore.py    # FAISS embeddings
│   └── rag.py            # RAG chain
├── requirements.txt
└── faiss_index/          # Persisted vector store
```

## Model Options

| Model | Parameters | Speed |
|-------|------------|-------|
| llama3.2:1b | 1B | Fast |
| llama3.2 | 3B | Medium |
