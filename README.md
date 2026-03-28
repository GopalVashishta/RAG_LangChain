# RAG Pipeline

Retrieval-Augmented Generation (RAG) project using LangChain, Sentence Transformers, and Groq LLM, with experiments in notebooks and a modular Python implementation in `src/`.

## What This Repo Contains

- Multi-format document loading (PDF, TXT, CSV, XLSX, DOCX, JSON)
- Text chunking and embedding generation (`all-MiniLM-L6-v2` by default)
- Vector search pipelines
  - Notebook workflow: ChromaDB (`data/vector_store/`)
  - `src/` workflow: FAISS (`faiss_store/` by default at runtime)
- RAG query answering with Groq (`simple`, `enhanced`, `advanced` patterns in notebook)

## Actual Project Structure

```text
RAG/
├── README.md
├── requirements.txt
├── data/
│   ├── pdf/
│   ├── text_files/
│   │   ├── machine_learning.txt
│   │   ├── python_intro.txt
│   │   └── sample1.txt
│   └── vector_store/
│       ├── chroma.sqlite3
│       └── f3828d17-731b-4d23-9234-12a3b06f22f2/
├── notebooks/
│   ├── document.ipynb
│   └── rag_pipeline.ipynb
└── src/
   ├── __init__.py
   ├── data_loader.py
   ├── embeddings.py
   ├── search.py
   └── vectorstore.py
```

## Pipeline Images

![Data Ingestion Pipeline](dingestion.png)
![Retrieval and Generation Pipeline](retrieve.png)

## RAG Flow (Based on `rag_pipeline.ipynb`)

```mermaid
flowchart LR
   A[Load Documents<br/>PDF/TXT/CSV/XLSX/DOCX/JSON] --> B[Split Into Chunks<br/>RecursiveCharacterTextSplitter]
   B --> C[Generate Embeddings<br/>SentenceTransformer]
   C --> D[(Vector Store<br/>ChromaDB in notebook<br/>FAISS in src)]

   E[User Query] --> F[Embed Query]
   F --> G[Similarity Search<br/>Top-K + Threshold]
   D --> G
   G --> H[Build Context From Retrieved Chunks]
   H --> I[Prompt Groq LLM]
   I --> J[Answer]

   J --> K[Enhanced Output: sources + confidence]
   J --> L[Advanced Output: history + optional summary]
```

## Quick Start

1. Create and activate environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add `.env` in project root

```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run notebook pipeline

```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

## Minimal `src/` Usage

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

docs = load_all_documents("data")

store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)

rag = RAGSearch(persist_dir="faiss_store")
answer = rag.search_and_summarize("What is attention mechanism?", top_k=3)
print(answer)
```

## Notes

- `notebooks/rag_pipeline.ipynb` demonstrates Simple, Enhanced, and Advanced RAG flows.
- `src/` modules implement a runnable FAISS-based pipeline.
- ChromaDB artifacts currently exist under `data/vector_store/` from notebook workflow.