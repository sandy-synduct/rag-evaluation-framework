import json
import faiss
import numpy as np
import streamlit as st
import re
from Bio import Entrez
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
from app.summarize_llm import LLMService
from app.ragas_eval import RagasEvaluator

# Load API keys from config
with open("config.json", "r") as f:
    config = json.load(f)

# Configure Entrez API (required by NCBI)
Entrez.email = "your_email@example.com"

# Load BioBERT model for embeddings
biobert_model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
bert_model = AutoModel.from_pretrained(biobert_model_name)

word_embedding_model = models.Transformer(biobert_model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

# SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Load FAISS index & document mappings
index = faiss.read_index("guidelines.index")
doc_indices = np.load("doc_indices.npy")

# Load guidelines JSON data
with open("processed_guidelines_v2_mini.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a mapping of index to guideline entry
guideline_mapping = {doc["guidelines_index"]: doc for doc in data}

# Initialize Evaluator
evaluator = RagasEvaluator()

# Streamlit UI
st.title("🔎 Clinical Guidelines Search")

# Choose between OpenAI and Gemini
model_choice = st.radio("Choose Summarization Model:", ("OpenAI (GPT-4)", "Google Gemini"))

# Load LLM Service based on choice
if model_choice == "OpenAI (GPT-4)":
    llm_service = LLMService(openai_api_key=config["openai_api_key"], model_name="gpt-4o-mini")
else:
    llm_service = LLMService(gemini_api_key=config["gemini_api_key"], model_name="gemini-2.0-flash-exp")

query = st.text_input("Enter your search query:")

# Function to normalize recency score
def normalize_recency(year, min_year=2000, max_year=2025):
    """Compute recency score in range [0, 1]."""
    if year is None or year < min_year:
        return 0.0
    if year > max_year:
        return 1.0
    return (year - min_year) / (max_year - min_year)

# Extract PubMed ID from URL
def extract_pubmed_id(pubmed_url):
    """Extract PubMed ID (PMID) from a given PubMed URL."""
    match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', pubmed_url)
    return match.group(1) if match else None

# Fetch MeSH terms from PubMed
def fetch_mesh_terms(pubmed_url):
    """Fetch MeSH terms for a given PubMed article using its URL."""
    pubmed_id = extract_pubmed_id(pubmed_url)
    if not pubmed_id:
        return []

    try:
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text")
        record = handle.read()
        handle.close()

        mesh_terms = re.findall(r'MH  - (.+)', record)
        return mesh_terms
    except Exception as e:
        print(f"❌ Error fetching MeSH terms: {e}")
        return []

# Compute MeSH Terms Score

import re

def compute_mesh_score(summary, mesh_terms):
    """Compute MeSH Score by checking presence of cleaned MeSH terms in the summary.
    
    - Splits MeSH terms at commas (','), hyphens ('-'), slashes ('/'), and spaces.
    - Only considers space as a separator if the next word has more than two letters.
    - Converts everything to lowercase for case-insensitive matching.
    - Counts how many MeSH terms appear in the summary.
    """
    if not mesh_terms:
        return None, []

    # Clean and expand MeSH terms
    cleaned_mesh_terms = set()
    for term in mesh_terms:
        # Split at comma, hyphen, slash
        split_terms = re.split(r'[,/-]', term)
        expanded_terms = []
        for t in split_terms:
            t = t.strip()
            if " " in t:
                words = t.split()
                for word in words:
                    if len(word) > 2 or not word.isalpha():
                        expanded_terms.append(word)
                # print(expanded_terms)
            else:
                expanded_terms.append(t)

        cleaned_mesh_terms.update([t.lower() for t in expanded_terms if t])

    # Tokenize summary
    summary_words = set(summary.lower().split())

    # Find matched terms
    matched_terms = [term for term in cleaned_mesh_terms if term in summary_words]

    # Compute score
    mesh_score = len(matched_terms) / len(cleaned_mesh_terms) if cleaned_mesh_terms else 0.0

    return round(mesh_score, 3), matched_terms

# Highlight MeSH terms in summary
def highlight_mesh_terms(summary, mesh_terms):
    """Highlight MeSH terms in the summary using tags."""
    for term in mesh_terms:
        summary = re.sub(f"(?i)({re.escape(term)})", r"**\1**", summary)
    return summary

# Search function
def search_guidelines(query, top_k=4):
    if not query:
        return [], []

    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results, abstracts = [], []
    for i, idx in enumerate(indices[0]):
        doc_id = doc_indices[idx]
        doc = guideline_mapping.get(doc_id)
        if doc:
            similarity_score = 1 / (1 + distances[0][i])
            recency_score = normalize_recency(doc.get("year"))

            results.append({
                "title": doc["title"],
                "year": doc.get("year", "N/A"),
                "recency_score": round(recency_score, 3),
                "citations": doc.get("score", 0),
                "PubMed": doc.get("PubMed"),
                "Link": doc.get("URL"),
                "Score": round(similarity_score, 3),
                "abstract": doc.get("abstract", "No abstract available")
            })
            abstracts.append(doc.get("abstract", "No abstract available"))

    return results, abstracts

# Execute search & summarization
if query:
    search_results, abstracts = search_guidelines(query, top_k=4)

    if search_results:
        summaries = llm_service.summarize_abstracts(abstracts)
        evaluation_results = evaluator.evaluate_summaries(query, abstracts, summaries)

        for i, result in enumerate(search_results):
            # Fetch MeSH terms if a PubMed link is available
            mesh_terms = fetch_mesh_terms(result["PubMed"]) if result["PubMed"] else []
            
            # Compute MeSH Score **ONLY IF MeSH terms exist**
            mesh_score, matched_terms = compute_mesh_score(summaries[i], mesh_terms) if mesh_terms else (None, [])
            
            # Highlight terms **ONLY IF MeSH terms exist**
            highlighted_summary = highlight_mesh_terms(summaries[i], matched_terms) if mesh_terms else summaries[i]

            st.write(f"### {result['title']}")
            st.write(f"**Year:** {result['year']}  |  **Citations:** {result['citations']}  |  **Recency Score:** {result['recency_score']}")
            st.write(f"**Abstract-Summary Similarity:** {evaluation_results.iloc[i]['semantic_similarity']:.3f}")  
            st.write(f"**Summary Relevancy to Query:** {evaluation_results.iloc[i]['answer_relevancy']:.3f}")
            
            # Display MeSH score **ONLY IF available**
            if mesh_score is not None:
                st.write(f"**MeSH Terms Score:** {mesh_score:.3f}")
                st.write(f"**Matched MeSH Terms:** {', '.join(matched_terms)}")

            st.write(f"**Summary:** {highlighted_summary}") 
            st.write(f"**PubMed:** {result['PubMed'] if result['PubMed'] else 'N/A'}")
            st.write(f"**Full Text:** {result['Link'] if result['Link'] else 'N/A'}")
            st.write("---")

    else:
        st.write("⚠️ No results found. Try a different query.")