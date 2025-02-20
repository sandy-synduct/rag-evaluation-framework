import re
from Bio import Entrez

class QMetrics:
    def __init__(self, pubmed_email):
        """Initialize with PubMed email required for Entrez API access."""
        Entrez.email = pubmed_email

    def normalize_recency(self, year, min_year=2015, max_year=2025):
        """Compute recency score in range [0, 1]."""
        if year is None or year < min_year:
            return 0.0
        if year > max_year:
            return 1.0
        return (year - min_year) / (max_year - min_year)

    def extract_pubmed_id(self, pubmed_url):
        """Extract PubMed ID (PMID) from a given PubMed URL."""
        match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', pubmed_url)
        return match.group(1) if match else None

    def fetch_mesh_terms(self, pubmed_url):
        """Fetch MeSH terms for a given PubMed article using its URL."""
        pubmed_id = self.extract_pubmed_id(pubmed_url)
        if not pubmed_id:
            return []

        try:
            handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="medline", retmode="text")
            record = handle.read()
            handle.close()
            mesh_terms = re.findall(r'MH  - (.+)', record)
            return mesh_terms
        except Exception as e:
            print(f"Error fetching MeSH terms: {e}")
            return []

    def compute_mesh_score(self, summary, mesh_terms):
        """Compute MeSH Score by checking presence of cleaned MeSH terms in the summary.
        
        - Splits MeSH terms at commas, hyphens, slashes, and spaces.
        - Only considers space as a separator if the next word has more than two letters.
        - Converts everything to lowercase for case-insensitive matching.
        - Counts how many MeSH terms appear in the summary.
        """
        if not mesh_terms:
            return None, []

        cleaned_mesh_terms = set()
        for term in mesh_terms:
            split_terms = re.split(r'[,/-]', term)
            expanded_terms = []
            for t in split_terms:
                t = t.strip()
                if " " in t:
                    words = t.split()
                    for word in words:
                        if len(word) > 2 or not word.isalpha():
                            expanded_terms.append(word)
                else:
                    expanded_terms.append(t)

            cleaned_mesh_terms.update([t.lower() for t in expanded_terms if t])

        summary_words = set(summary.lower().split())
        matched_terms = [term for term in cleaned_mesh_terms if term in summary_words]

        mesh_score = len(matched_terms) / len(cleaned_mesh_terms) if cleaned_mesh_terms else 0.0

        return round(mesh_score, 3), matched_terms
    
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