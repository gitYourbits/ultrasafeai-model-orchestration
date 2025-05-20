import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

class VectorStore:
    def __init__(self, persist_directory: str = "rag/chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection("financial_docs")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_documents(self, docs: List[Dict]):
        self.logger.info(f"Adding {len(docs)} documents to vector store.")
        texts = [doc["text"] for doc in docs]
        ids = [doc["id"] for doc in docs]
        metadatas = [doc.get("metadata", {}) for doc in docs]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        self.logger.info("Documents added successfully.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        self.logger.info(f"Searching vector store for query: '{query[:50]}...' (top_k={top_k})")
        embedding = self.embedder.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        self.logger.info(f"Found {len(hits)} results.")
        return hits 