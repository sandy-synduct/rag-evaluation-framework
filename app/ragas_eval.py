from datasets import Dataset
from ragas.metrics import answer_similarity, answer_relevancy, summarization_score
from ragas import evaluate
from ragas import SingleTurnSample 
from ragas.metrics import ResponseRelevancy

class RagasEvaluator:
    def __init__(self):
        pass
    def clean_data(self, text_list):
        """Ensure all elements are strings and replace None with empty strings."""
        return [str(text).strip() if text else "" for text in text_list]

    def evaluate_summaries(self, query, abstracts, summaries):
        """Evaluate summaries using answer similarity and response relevancy."""
        abstracts = self.clean_data(abstracts)
        summaries = self.clean_data(summaries)

        if not abstracts or not summaries or len(abstracts) != len(summaries):
            raise ValueError("Mismatch in abstracts and summaries length or empty inputs.")

        queries = [query] * len(abstracts)  # Ensure queries match dataset size

        # ** Answer Similarity Evaluation**
        data_samples = {
            "question": queries,
            "answer": summaries,
            "ground_truth": abstracts,
        }
        dataset = Dataset.from_dict(data_samples)

        try:
            score = evaluate(dataset, metrics=[answer_similarity,answer_relevancy])
            # answer_similarity_scores = score.to_pandas()
            # score = evaluate(dataset,metrics=[answer_relevancy])
        except Exception as e:
            print(f"Answer Similarity Evaluation failed: {e}")
            score = None
        return score.to_pandas()