from popgen.population import DiversityEvaluator


def test_diversity_evaluator_single_text_returns_zero():
    evaluator = DiversityEvaluator(embedding_model="dummy-model")
    metrics = evaluator.compute_instance_metric({"texts": ["one behavior"]})

    assert metrics["semantic_diversity"] == 0.0
    assert metrics["lexical_diversity"] == 0.0
    assert metrics["semantic_similarity"] is None
    assert metrics["lexical_similarity"] is None
