from fastapi import FastAPI, Query
from pydantic import BaseModel
import faiss
import numpy as np
import json
from config import config
from faiss_search import FAISSSearch
from summarize_llm import LLMService
from ragas_eval import RagasEvaluator
from other_eval import QMetrics


# Initialize FastAPI
app = FastAPI(title="Clinical Guidelines Evaluation API")

# Load FAISS Index
faiss_search = FAISSSearch()

# Initialize Evaluators
ragas_evaluator = RagasEvaluator()
mesh_evaluator = QMetrics(config.PUBMED_EMAIL)


# Initialize LLM Services
llm_service_gpt = LLMService(openai_api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini")
llm_service_gemini = LLMService(gemini_api_key=config.GEMINI_API_KEY, model_name="gemini-2.0-flash-exp")


# Request model for summarization
class QueryRequest(BaseModel):
    query: str
    model: str = "OpenAI"  # Choose between "OpenAI" or "Gemini"
    top_k: int = 4

# API Endpoint: Search Clinical Guidelines
@app.post("/search/")
async def search_guidelines(request: QueryRequest, p: float = 0.7):
    """
    Search and rank guidelines based on similarity and recency.
    
    - p (default=0.7): Weight for similarity.
    - (1-p) is used as the weight for recency.
    """
    results, abstracts = faiss_search.search(request.query, request.top_k)
    if not results:
        return {"message": "No results found", "query": request.query}
    p = max(0, min(p, 1))
    ranked_results = QMetrics.rank_results(results, p)
    return {"query": request.query, "results": ranked_results}

# API Endpoint: Summarize Retrieved Abstracts
@app.post("/summarize/")
async def summarize_guidelines(request: QueryRequest):
    results, abstracts = faiss_search.search(request.query, request.top_k)
    
    if not results:
        return {"message": "No results found", "query": request.query}

    llm_service = llm_service_gpt if request.model == "OpenAI" else llm_service_gemini
    summaries = llm_service.summarize_abstracts(abstracts)

    return {"query": request.query, "summaries": summaries}

# API Endpoint: Evaluate Summaries (Semantic & MeSH)
@app.post("/evaluate/")
async def evaluate_guidelines(request: QueryRequest):
    results, abstracts = faiss_search.search(request.query, request.top_k)

    if not results:
        return {"message": "No results found", "query": request.query}

    llm_service = llm_service_gpt if request.model == "OpenAI" else llm_service_gemini
    summaries = llm_service.summarize_abstracts(abstracts)

    # Evaluate semantic similarity & relevancy
    evaluation_results = ragas_evaluator.evaluate_summaries(request.query, abstracts, summaries)

    # Evaluate MeSH Score (only if PubMed links are available)
    # print(results)
    other_scores = []
    for i, result in enumerate(results):
        mesh_terms = mesh_evaluator.fetch_mesh_terms(result["PubMed"]) if result.get("PubMed") else []
        mesh_score, matched_terms = mesh_evaluator.compute_mesh_score(summaries[i], mesh_terms) if mesh_terms else (None, [])
        recency_score = mesh_evaluator.normalize_recency(result['year'])
        other_scores.append({"score": mesh_score, "matched_terms": matched_terms,"recency_score":recency_score})

    # Construct final response
    print(other_scores)
    evaluated_results = []
    for i, result in enumerate(results):
        evaluated_results.append({
            "title": result["title"],
            "year": result["year"],
            "citations": result["citations"],
            "recency_score": other_scores[i]["recency_score"],
            "abstract_summary_similarity": evaluation_results.iloc[i]["semantic_similarity"],
            "summary_relevancy": evaluation_results.iloc[i]["answer_relevancy"],
            "mesh_score": other_scores[i]["score"],
            "matched_mesh_terms": other_scores[i]["matched_terms"],
            "summary": summaries[i],
            "PubMed": result.get("PubMed", "N/A"),
            "Full Text": result.get("Link", "N/A")
        })

    return {"query": request.query, "evaluated_results": evaluated_results}