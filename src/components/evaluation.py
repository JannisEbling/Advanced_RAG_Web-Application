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

from src.components.llms import get_llm_model

with open("data_samples.json", "r", encoding="utf-8") as f:
    data_samples = json.load(f)


# Convert contexts to list of strings (if necessary)
data_samples["contexts"] = [list(context) for context in data_samples["contexts"]]

dataset = Dataset.from_dict(data_samples)

# Evaluate using Ragas with the specified metrics
metrics = [
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity,
]
llm = get_llm_model("base_azure")
score = evaluate(dataset, metrics=metrics, llm=llm)

# Print results and explanations
results_df = score.to_pandas()
print(results_df)
