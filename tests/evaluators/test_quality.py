from unittest.mock import MagicMock, patch

from popgen.population import BehaviorQualityEvaluator, BehaviorQualityResult, RubricScore


def _mock_quality_response(relevance: int, realism: int, specificity: int) -> MagicMock:
    result = BehaviorQualityResult(
        relevance=RubricScore(score=relevance, reasoning="rel"),
        realism=RubricScore(score=realism, reasoning="real"),
        specificity=RubricScore(score=specificity, reasoning="spec"),
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = result.model_dump_json()
    return mock_response


def test_quality_evaluator_returns_scores():
    evaluator = BehaviorQualityEvaluator(judge_model="gpt-5-nano")
    with patch("popgen.population.evaluation.quality.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_quality_response(4, 3, 2)

        metrics = evaluator.compute_instance_metric(
            {
                "behavior": "Do X",
                "environment_description": "Env desc",
            }
        )

    assert metrics["relevance"] == 4
    assert metrics["realism"] == 3
    assert metrics["specificity"] == 2
    assert metrics["total"] == 9
    assert metrics["error"] is None
