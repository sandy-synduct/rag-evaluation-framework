import faiss
import numpy as np
import json
from config import config
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel

class FAISSSearch:
    def __init__(self):
        """
        Initialize FAISS search engine by loading index, document indices, and guideline metadata.
        """
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        self.doc_indices = np.load(config.DOC_INDICES_PATH)

        with open(config.GUIDELINES_JSON, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Create a mapping of index to guideline entry
        self.guideline_mapping = {doc["guidelines_index"]: doc for doc in self.data}

        # Store only document indices where year ≤ 2015
        self.pre_2015_indices = np.array([
                    idx for idx in range(len(self.doc_indices)) 
                    if isinstance(self.guideline_mapping.get(self.doc_indices[idx], {}).get("year"), int) 
                    and self.guideline_mapping[self.doc_indices[idx]]["year"] >= 2015
                ])
        
        # Load BioBERT model
        self.biobert_model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.biobert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.biobert_model_name)

        word_embedding_model = models.Transformer(self.biobert_model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    def normalize_recency(self, year, min_year=2015, max_year=2025):
        """Compute recency score in range [0, 1]."""
        if year is None or year < min_year:
            return 0.0
        if year > max_year:
            return 1.0
        return (year - min_year) / (max_year - min_year)
    
    def search(self, query: str, top_k: int = 30, filter_pre_2015: bool = False):
        if not query:
            return [], []

        query_embedding = self._encode_query(query).reshape(1, -1)

        # Select subset of indices based on filtering condition
        if filter_pre_2015:
            search_indices = self.pre_2015_indices
            filtered_index = faiss.IndexIDMap2(self.index)  # Create a subset index
            filtered_index.add_with_ids(self.index.reconstruct_n(search_indices, len(search_indices)), search_indices)
        else:
            filtered_index = self.index  # Use full index

        # Perform FAISS search only on the selected subset
        distances, indices = filtered_index.search(query_embedding, top_k)

        results, abstracts = [], []
        for i, idx in enumerate(indices[0]):
            doc_id = self.doc_indices[idx]
            doc = self.guideline_mapping.get(doc_id)

            if doc:
                recency_score = self.normalize_recency(doc.get("year", "N/A"))
                similarity_score = 1 / (1 + distances[0][i])
                results.append({
                    "title": doc["title"],
                    "year": doc.get("year", "N/A"),
                    "citations": doc.get("score", 0),
                    "PubMed": doc.get("PubMed"),
                    "Link": doc.get("URL"),
                    "Score": round(similarity_score, 3),
                    "recency_score": recency_score,
                    "abstract": doc.get("abstract", "No abstract available")
                })
                abstracts.append(doc.get("abstract", "No abstract available"))

        return results, abstracts


    def _encode_query(self, query: str):
        """
        Encode the query into an embedding using BioBERT.
        """
        return self.model.encode(query, convert_to_numpy=True)
    

    def rank_results(results, p=0.7):
        """
        Rank retrieved guidelines based on a weighted combination of similarity and recency.

        - p: weight for similarity (0 to 1)
        - 1-p: weight for recency (ensuring the sum is 1)
        """
        alpha = p  # Similarity weight
        beta = 1 - p  # Recency weight
        
        for result in results:
            similarity = result["Score"]  # FAISS similarity score
            recency = result["recency_score"]  # Already normalized between 0 and 1
            
            # Compute weighted score
            result["final_score"] = alpha * similarity + beta * recency

        # Sort results by final score (higher is better)
        ranked_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        return ranked_results