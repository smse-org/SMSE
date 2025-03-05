import torch

from smse.logging import get_logger
from smse.recipes.metrics import map, mrr, ndcg, precision, recall
from smse.types import Modality

logger = get_logger(__name__)


def text_evaluator_st_example() -> None:
    """Example of using a SentenceTransformer model with detailed ranking debugging."""
    from smse.benchmarks.text_evaluator import TextEvaluator
    from smse.models.sentence_transformers import SentenceTransformerTextModel

    logger.info("=== SentenceTransformer Model Evaluation ===")
    model = SentenceTransformerTextModel("all-MiniLM-L6-v2")

    # Small hard-coded dataset simulating MSMarco-like queries and answers
    sentences = [
        # Each tuple is (query, relevant_answer)
        ("how to improve mental health", "tips for maintaining good mental health"),
        (
            "best practices for software development",
            "key principles of effective software engineering",
        ),
        (
            "machine learning introduction",
            "beginner's guide to machine learning concepts",
        ),
        (
            "nutrition for athletes",
            "optimal diet and nutrition for professional athletes",
        ),
        (
            "renewable energy sources",
            "comprehensive overview of sustainable energy technologies",
        ),
        (
            "benefits of meditation",
            "how meditation enhances mental and physical well-being",
        ),
        (
            "top programming languages",
            "most popular and widely used programming languages today",
        ),
        (
            "best exercises for weight loss",
            "effective workouts for shedding extra pounds",
        ),
        ("how to cook pasta", "step-by-step guide to making perfect pasta dishes"),
        (
            "importance of financial planning",
            "why managing personal finances effectively is crucial",
        ),
        (
            "how to learn a new language",
            "strategies for mastering a foreign language quickly",
        ),
        ("best time to visit Japan", "seasonal travel guide for exploring Japan"),
        (
            "how to stay productive while working from home",
            "tips for maintaining efficiency and focus in remote work",
        ),
    ]

    metrics = [
        ndcg.NDCG(),
        recall.Recall(),
        mrr.MRR(),
        precision.Precision(),
        map.MAP(),
    ]

    # Create the text evaluator
    evaluator = TextEvaluator(model, metrics)

    try:
        # Unpack dataset
        queries_raw, relevant_answers_raw = zip(*sentences)
        queries = list(queries_raw)
        relevant_answers = list(relevant_answers_raw)

        # Encode queries and answers
        queries_embedded = model.encode({Modality.TEXT: queries})[Modality.TEXT]
        relevant_answers_embedded = model.encode({Modality.TEXT: relevant_answers})[
            Modality.TEXT
        ]

        # Calculate full similarity matrix (all queries vs all answers)
        similarity_matrix = model.similarity(
            queries_embedded, relevant_answers_embedded
        )

        # Convert similarity matrix to rankings
        predictions = evaluator.cosine_to_rankings(similarity_matrix)

        # Detailed logging of similarity and rankings
        logger.info("\n=== Similarity Matrix ===")
        logger.info(similarity_matrix)

        logger.info("\n=== Rankings Matrix ===")
        logger.info(predictions)

        # Detailed logging of conversion process
        logger.info("\n=== Detailed Ranking Conversion ===")
        for i in range(len(sentences)):
            logger.info(f"\nQuery {i}: '{queries[i]}'")
            logger.info("Similarities and Rankings:")

            # Get sorted indices of similarities in descending order
            sorted_indices = similarity_matrix[i].argsort(descending=True)

            for rank, doc_index in enumerate(sorted_indices, 1):
                logger.info(
                    f"  Rank {rank}: Document {doc_index.item()} ('{relevant_answers[doc_index.item()]}'):"
                )
                logger.info(
                    f"    Similarity: {similarity_matrix[i, doc_index].item():.4f}"
                )
                logger.info(f"    Predicted Rank: {predictions[i, doc_index].item()}")

        # Create ground truth (assuming diagonal as most relevant)
        ground_truth = torch.zeros_like(predictions)
        for i in range(len(sentences)):
            ground_truth[i, i] = 1.0  # Highest relevance to corresponding document

        results = evaluator.run(sentences)

        # Log results
        logger.info("\n=== Evaluation Results ===")
        for metric_name, metric_value in results.items():
            logger.info(f"{metric_name}: {metric_value}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
