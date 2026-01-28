from popgen.population import DiversityEvaluator


def test_diversity_evaluator_single_text_returns_zero():
    evaluator = DiversityEvaluator(embedding_model="dummy-model")
    metrics = evaluator.compute_instance_metric({"texts": ["one behavior"]})

    assert metrics["semantic_diversity"] == 0.0
    assert metrics["lexical_diversity"] == 0.0
    assert len(metrics["semantic_similarity_matrix"]) == 0
    assert len(metrics["lexical_similarity_matrix"]) == 0