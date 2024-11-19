import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_recall,
    faithfulness,
)

from src.components.llm_factory import LLMFactory


def evaluate_rag_pipeline(data_samples_path: str = "data_samples.json"):
    """
    Evaluate the RAG pipeline using Ragas metrics.

    Args:
        data_samples_path (str): Path to the JSON file containing evaluation data samples
    
    Returns:
        DataFrame: Evaluation results
    """
    try:
        # Load evaluation data
        with open(data_samples_path, "r", encoding="utf-8") as f:
            data_samples = json.load(f)

        # Convert contexts to list of strings (if necessary)
        data_samples["contexts"] = [list(context) for context in data_samples["contexts"]]

        # Create dataset
        dataset = Dataset.from_dict(data_samples)

        # Initialize metrics
        metrics = [
            answer_correctness,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_similarity,
        ]

        # Initialize LLM for evaluation
        llm = LLMFactory(provider="azure")

        # Evaluate using Ragas
        score = evaluate(dataset, metrics=metrics, llm=llm)

        # Convert results to pandas DataFrame
        results_df = score.to_pandas()
        return results_df

    except Exception as e:
        logger.error("Failed to evaluate RAG pipeline", exc_info=True)
        raise MultiAgentRAGException("Failed to evaluate RAG pipeline") from e


if __name__ == "__main__":
    results = evaluate_rag_pipeline()
    print("\nEvaluation Results:")
    print(results)
