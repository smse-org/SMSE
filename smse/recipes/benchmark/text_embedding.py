from smse.logging import get_logger
from smse.benchmarks import text_evaluator
from smse.types import Modality
from smse.benchmarks.text_evaluator import Evaluator
from smse.models.sentence_transformers import SentenceTransformerTextModel
from smse.recipes.metrics import spearman , pearson, ndcg, recall, topk, mrr

import numpy as np

logger = get_logger(__name__)

def text_model_st_example() -> None:
    """Example of using a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    logger.info("=== SentenceTransformer Model ===")
    model = SentenceTransformerTextModel("all-MiniLM-L6-v2")

    sentences = [
        "The cat is sitting on the mat.",
        "A dog runs through the park.",
        "The weather is nice today.",
        "I enjoy reading science fiction books.",
        "The train arrives at the station on time.",
        "She plays the piano beautifully.",
        "The restaurant serves delicious food.",
        "He is studying computer science at the university.",
        "The mountains are covered with snow.",
        "Children are playing in the garden."
    ]

    metrics = [
        spearman.SpearmanCorrelation(),
        pearson.PearsonCorrelation(),
        ndcg.NDCG(),
        recall.Recall(),
        topk.TopK(),
        mrr.MRR()
    ]

    logger.info("Loading evaluator")
    evaluator = Evaluator(model, metrics)

    logger.info("Running evaluation")
    evaluator.run(sentences, sentences)

    results = evaluator.get_results()
    logger.info(f"Evaluation Results:\n{results}")