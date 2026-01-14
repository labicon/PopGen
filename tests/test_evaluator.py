import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from popgen.population import (
    BehaviorEvaluator,
    EvaluationResult,
    Behavior
)


class TestEvaluationResultModel:
    def test_evaluation_creation(self):
        evaluation = EvaluationResult(evaluation=True, reasoning="Agent followed behavior")
        assert evaluation.evaluation is True
        assert evaluation.reasoning == "Agent followed behavior"

    def test_evaluation_from_dict(self):
        data = {"evaluation": False, "reasoning": "Agent deviated"}
        evaluation = EvaluationResult.model_validate(data)
        assert evaluation.evaluation is False
        assert evaluation.reasoning == "Agent deviated"

    def test_evaluation_to_dict(self):
        evaluation = EvaluationResult(evaluation=True, reasoning="Good")
        data = evaluation.model_dump()
        assert data == {"evaluation": True, "reasoning": "Good"}

    def test_evaluation_missing_field_raises(self):
        with pytest.raises(ValueError):
            EvaluationResult(evaluation=True)


class TestBehaviorEvaluator:
    def test_init_custom_model(self):
        evaluator = BehaviorEvaluator(model="gpt-4-turbo")
        assert evaluator.model == "gpt-4-turbo"


def _mock_evaluation_response(evaluation: bool, reasoning: str) -> MagicMock:
    """Helper to create a mock LLM response with JSON content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = EvaluationResult(
        evaluation=evaluation, reasoning=reasoning
    ).model_dump_json()
    return mock_response


class TestEvaluateRollouts:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def evaluator(self):
        return BehaviorEvaluator(model="gpt-5-nano")

    @pytest.fixture
    def sample_rollout_states(self):
        return [[{"position": [0, 0], "velocity": [1, 0]}, {"position": [1, 0], "velocity": [1, 0]}]]

    @pytest.fixture
    def sample_rollout_images(self):
        return [[np.zeros((64, 64, 3), dtype=np.uint8)]]

    def test_evaluate_rollouts_with_behavior_object(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "Followed correctly")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A red circle",
            rollouts_states=sample_rollout_states,
        )

        assert result.evaluation is True
        assert result.reasoning == "Followed correctly"
        mock_completion.assert_called_once()

    def test_evaluate_rollouts_without_images(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(False, "Did not follow")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A red circle",
            rollouts_states=sample_rollout_states,
            rollouts_images=None,
        )

        assert result.evaluation is False
        assert result.reasoning == "Did not follow"

    def test_evaluate_rollouts_includes_behavior_in_prompt(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Aggressive", description="Always attacks first")
        evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Attacker",
            ego_visual_description="A red triangle",
            rollouts_states=sample_rollout_states,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        # user_content is a list of content blocks
        text_content = "".join(
            block["text"] for block in user_content if block["type"] == "text"
        )
        assert "Aggressive" in text_content
        assert "Always attacks first" in text_content

    def test_evaluate_rollouts_uses_correct_response_format(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A circle",
            rollouts_states=sample_rollout_states,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == EvaluationResult

    def test_evaluate_rollouts_returns_failure_on_exception(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.side_effect = Exception("API error")

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A circle",
            rollouts_states=sample_rollout_states,
        )

        assert result.evaluation is False
        assert "Evaluation failed" in result.reasoning


class TestEvaluateRolloutsWithImages:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def mock_encode_images(self):
        with patch("popgen.population.evaluator.encode_images_to_base64") as mock:
            mock.return_value = ["base64encodedimage"]
            yield mock

    @pytest.fixture
    def evaluator(self):
        return BehaviorEvaluator(model="gpt-5-nano")

    @pytest.fixture
    def sample_rollout_states(self):
        return [[{"position": [0, 0]}, {"position": [1, 0]}]]

    @pytest.fixture
    def sample_rollout_images(self):
        return [[np.zeros((64, 64, 3), dtype=np.uint8)]]

    def test_evaluate_with_images_success(self, mock_completion, mock_encode_images, evaluator, sample_rollout_states, sample_rollout_images):
        mock_completion.return_value = _mock_evaluation_response(True, "Visual confirms behavior")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A red circle",
            rollouts_states=sample_rollout_states,
            rollouts_images=sample_rollout_images,
        )

        assert result.evaluation is True
        assert result.reasoning == "Visual confirms behavior"
        mock_encode_images.assert_called_once()

    def test_evaluate_with_images_includes_images_in_content(self, mock_completion, mock_encode_images, evaluator, sample_rollout_states, sample_rollout_images):
        mock_encode_images.return_value = ["base64data"]
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A circle",
            rollouts_states=sample_rollout_states,
            rollouts_images=sample_rollout_images,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        assert isinstance(user_content, list)
        image_entries = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_entries) == 1
        assert "base64data" in image_entries[0]["image_url"]["url"]

    def test_evaluate_with_multiple_rollouts(self, mock_completion, mock_encode_images, evaluator):
        mock_encode_images.return_value = ["img1", "img2"]
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        rollouts_states = [
            [{"step": 0}, {"step": 1}],
            [{"step": 0}, {"step": 1}],
        ]
        rollouts_images = [
            [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)],
            [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)],
        ]

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_rollouts(
            behavior=behavior,
            ego_name="Agent1",
            ego_visual_description="A circle",
            rollouts_states=rollouts_states,
            rollouts_images=rollouts_images,
        )

        assert result.evaluation is True
        assert mock_encode_images.call_count == 2  # Once per rollout
