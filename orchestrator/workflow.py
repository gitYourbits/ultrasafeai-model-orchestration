import os
import sys
import logging
from langgraph.graph import StateGraph, END
from agents.document_parser import DocumentParserAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_generator import ReportGeneratorAgent
from rag.vector_store import VectorStore
from rag.reranker import Reranker


def parse_document(state: dict) -> dict:
    logging.info("Step: parse_document")
    try:
        parser = DocumentParserAgent()
        state["extracted_text"] = parser.parse_pdf(state["pdf_path"])
    except Exception as e:
        state["error"] = f"Document parsing failed: {str(e)}"
        logging.error(state["error"])
    return state

def retrieve_context(state: dict) -> dict:
    logging.info("Step: retrieve_context")
    if state.get("error") or not state.get("extracted_text"):
        return state
    try:
        vector_store = VectorStore()
        reranker = Reranker()
        query = state["extracted_text"][:500]
        retrieved = vector_store.search(query, top_k=8)
        reranked = reranker.rerank(query, retrieved, top_k=3)
        state["retrieved_context"] = reranked
    except Exception as e:
        state["error"] = f"Context retrieval failed: {str(e)}"
        logging.error(state["error"])
    return state

def analyze_text(state: dict) -> dict:
    logging.info("Step: analyze_text")
    if state.get("error") or not state.get("extracted_text"):
        return state
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        agent = AnalysisAgent(api_key, base_url)
        context = "\n\n".join([doc["text"] for doc in (state.get("retrieved_context") or [])])
        input_text = state["extracted_text"]
        if context:
            input_text = f"Relevant Context:\n{context}\n\nReport Text:\n{state['extracted_text']}"
        state["metrics"] = agent.analyze(input_text)
    except Exception as e:
        state["error"] = f"Analysis failed: {str(e)}"
        logging.error(state["error"])
    return state

def generate_report(state: dict) -> dict:
    logging.info("Step: generate_report")
    if state.get("error") or not state.get("metrics"):
        return state
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        agent = ReportGeneratorAgent(api_key, base_url)
        state["report"] = agent.generate_report(state["metrics"])
    except Exception as e:
        state["error"] = f"Report generation failed: {str(e)}"
        logging.error(state["error"])
    return state

def build_workflow():
    graph = StateGraph(dict)
    graph.add_node("parse_document", parse_document)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("analyze_text", analyze_text)
    graph.add_node("generate_report", generate_report)
    graph.add_edge("parse_document", "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_text")
    graph.add_edge("analyze_text", "generate_report")
    graph.add_edge("generate_report", END)
    graph.set_entry_point("parse_document")
    return graph.compile()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if len(sys.argv) < 2:
        print("Usage: python orchestrator/workflow.py <path_to_pdf>")
        exit(1)
    workflow = build_workflow()

    state = {
        "pdf_path": sys.argv[1],
        "extracted_text": None,
        "metrics": None,
        "report": None,
        "error": None,
        "retrieved_context": None
    }

    result = workflow.invoke(state)

    if result.get("error"):
        logging.error(f"Workflow failed: {result['error']}")
        print(f"Workflow failed: {result['error']}")
    else:
        logging.info("Workflow completed successfully. Final report generated.")
        print("\n===== FINAL REPORT =====\n")
        print(result["report"])
