# app/evaluation/ragas_runner.py
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_similarity, context_precision
from datasets import Dataset
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings

def evaluate_samples(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    dataset = Dataset.from_list(samples)

    # Local LLM and embeddings
    llm = ChatOllama(model="llama3.1")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_similarity, context_precision],
        llm=llm,
        embeddings=embeddings
    )
    score_list = results.scores  # List of dicts
    aggregated = {}
    for metric_name in score_list[0].keys():
        values = [sample[metric_name] for sample in score_list if sample[metric_name] is not None]
        aggregated[metric_name] = sum(values) / len(values) if values else 0.0

    return aggregated


#python -m streamlit run .\app\evaluation\dashboard.py
