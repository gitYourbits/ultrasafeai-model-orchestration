# Multi-Agent System for Financial Report Analysis

## Overview
This project implements a modular multi-agent system using LangGraph for analyzing financial reports. The system consists of three specialized agents:

1. **Document Parser Agent**: Extracts text from financial PDF documents.
2. **Analysis Agent**: Identifies key financial metrics and trends using an OpenAI-compatible LLM.
3. **Report Generator Agent**: Generates a human-readable summary of the findings.

## Directory Structure
```
agents/
  document_parser.py
  analysis_agent.py
  report_generator.py
orchestrator/
  workflow.py
rag/
  vector_store.py
  reranker.py
requirements.txt
README.md
```

## Agent Details

### 1. Document Parser Agent
- **Location**: `agents/document_parser.py`
- **Function**: Extracts text from PDF files using `pdfplumber`.
- **Usage**:
  ```bash
  python agents/document_parser.py <path_to_pdf>
  ```

### 2. Analysis Agent
- **Location**: `agents/analysis_agent.py`
- **Function**: Uses an OpenAI-compatible LLM to extract key metrics and trends from text.
- **Environment Variables**:
  - `OPENAI_API_KEY`: Your API key
  - `OPENAI_BASE_URL`: The base URL for the OpenAI-compatible API
- **Usage**:
  ```bash
  python agents/analysis_agent.py <path_to_text_file>
  ```

### 3. Report Generator Agent
- **Location**: `agents/report_generator.py`
- **Function**: Uses an LLM to generate a summary from extracted metrics.
- **Environment Variables**:
  - `OPENAI_API_KEY`: Your API key
  - `OPENAI_BASE_URL`: The base URL for the OpenAI-compatible API
- **Usage**:
  ```bash
  python agents/report_generator.py <path_to_metrics_json>
  ```

## Orchestrator Workflow (LangGraph)
- **Location**: `orchestrator/workflow.py`
- **Function**: Orchestrates the three agents in a workflow using LangGraph. Takes a PDF, extracts text, retrieves relevant context, analyzes it, and generates a report.
- **Usage**:
  ```bash
  python orchestrator/workflow.py <path_to_pdf>
  ```
- **Architecture**:
  1. **parse_document**: Extracts text from the PDF.
  2. **retrieve_context**: Retrieves and reranks relevant financial information from the vector store.
  3. **analyze_text**: Extracts key metrics and trends from the text, using retrieved context.
  4. **generate_report**: Produces a human-readable summary.
  5. Handles errors at each step and stops if any occur.

## RAG Components

### Vector Store (`rag/vector_store.py`)
- Uses ChromaDB and SentenceTransformers to store and retrieve financial information as embeddings.
- **Key Methods:**
  - `add_documents(docs)`: Add documents (with id, text, metadata) to the store.
  - `search(query, top_k)`: Retrieve top-k relevant documents for a query.
- **Example Usage:**
  ```python
  from rag.vector_store import VectorStore
  store = VectorStore()
  store.add_documents([
      {"id": "doc1", "text": "Annual report 2022: Revenue $1B...", "metadata": {"year": 2022}},
      # ... more docs ...
  ])
  results = store.search("Q4 revenue", top_k=3)
  ```

### Reranker (`rag/reranker.py`)
- Uses a cross-encoder model to rerank retrieved documents for relevance to the query.
- **Key Methods:**
  - `rerank(query, docs, top_k)`: Returns top-k docs sorted by rerank score.

### Context Injection
- Before analysis, the workflow retrieves and reranks relevant context from the vector store.
- The top reranked context is prepended to the extracted text and provided to the Analysis Agent.
- This ensures the agent has access to both the current document and relevant prior knowledge, improving accuracy and depth of analysis.

## How RAG Enhances the System
- **Improved Relevance:** The system can reference prior financial documents, industry benchmarks, or regulatory guidelines, making analysis more robust.
- **Contextual Awareness:** Agents are not limited to the current document—they leverage a knowledge base for deeper insights.
- **Modularity:** The RAG components can be updated or expanded independently.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables for API access:
   ```bash
   set OPENAI_API_KEY=your_api_key
   set OPENAI_BASE_URL=api_provider_base_url
   ```

## Example Implementation
- Place sample financial reports in `data/sample_reports/`.
- Add them to the vector store using the provided API.
- Run the orchestrator workflow on a new PDF to see context-aware analysis and reporting.

---
For more details, see each agent's source code and docstrings. 
