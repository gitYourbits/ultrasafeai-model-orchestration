from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    def rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        self.logger.info(f"Reranking {len(docs)} documents for query: '{query[:50]}...' (top_k={top_k})")
        pairs = [[query, doc["text"]] for doc in docs]
        scores = self.model.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        self.logger.info(f"Reranking complete. Returning top {top_k}.")
        return docs[:top_k] 