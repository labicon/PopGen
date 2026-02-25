import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from popgen.population import BehaviorAdherenceEvaluator, BehaviorAdherenceResult, Behavior
from popgen.utils.config import LitellmConfig


def _mock_evaluation_response(evaluation: bool, reasoning: str, feedback: str = "No feedback.") -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = None
    mock_response.choices[0].message.content = BehaviorAdherenceResult(
        evaluation=evaluation, reasoning=reasoning, feedback=feedback
    ).model_dump_json()
    return mock_response


class TestBehaviorAdherenceResultModel:
    def test_evaluation_creation(self):
        evaluation = BehaviorAdherenceResult(evaluation=True, reasoning="Agent followed behavior", feedback="Keep it up.")
        assert evaluation.evaluation is True
        assert evaluation.reasoning == "Agent followed behavior"

    def test_evaluation_from_dict(self):
        data = {"evaluation": False, "reasoning": "Agent deviated", "feedback": "Change strategy."}
        evaluation = BehaviorAdherenceResult.model_validate(data)
        assert evaluation.evaluation is False
        assert evaluation.reasoning == "Agent deviated"

    def test_evaluation_to_dict(self):
        evaluation = BehaviorAdherenceResult(evaluation=True, reasoning="Good", feedback="Nothing to change.")
        data = evaluation.model_dump()
        assert data == {"evaluation": True, "reasoning": "Good", "feedback": "Nothing to change."}

    def test_evaluation_missing_field_raises(self):
        with pytest.raises(ValueError):
            BehaviorAdherenceResult(evaluation=True)


class TestBehaviorAdherenceEvaluator:
    def test_init_custom_model(self):
        config = LitellmConfig(model_name="gpt-4-turbo")
        evaluator = BehaviorAdherenceEvaluator(model_config=config)
        assert evaluator.model_config.model_name == "gpt-4-turbo"


class TestComputeInstanceMetric:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluation.adherence.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def evaluator(self):
        config = LitellmConfig(model_name="gpt-5-nano")
        return BehaviorAdherenceEvaluator(model_config=config, num_workers=1)

    @pytest.fixture
    def sample_rollout_states(self):
        return [{"position": [0, 0], "velocity": [1, 0]}, {"position": [1, 0], "velocity": [1, 0]}]

    @pytest.fixture
    def sample_rollout_images(self):
        return [np.zeros((64, 64, 3), dtype=np.uint8)]

    def test_compute_instance_metric_with_behavior_object(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "Followed correctly")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": [],
            }
        )

        assert result["adherence"] == 1.0
        assert "Followed correctly" in result["reasoning"]
        mock_completion.assert_called_once()

    def test_compute_instance_metric_without_images(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(False, "Did not follow")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": [],
            }
        )

        assert result["adherence"] == 0.0
        assert "Did not follow" in result["reasoning"]

    def test_compute_instance_metric_includes_behavior_in_prompt(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Aggressive", description="Always attacks first")
        evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": [],
            }
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        text_content = "".join(
            block["text"] for block in user_content if block["type"] == "text"
        )
        assert "Aggressive" in text_content
        assert "Always attacks first" in text_content

    def test_compute_instance_metric_uses_correct_response_format(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": [],
            }
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == BehaviorAdherenceResult

    def test_compute_instance_metric_returns_failure_on_exception(self, mock_completion, evaluator, sample_rollout_states):
        mock_completion.side_effect = Exception("API error")

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
            }
        )

        assert result["adherence"] == 0.0
        assert "Evaluation failed" in result["reasoning"]


class TestComputeInstanceMetricWithImages:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluation.adherence.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def mock_encode_images(self):
        with patch("popgen.population.evaluation.adherence.encode_images_to_base64") as mock:
            mock.return_value = ["base64encodedimage"]
            yield mock

    @pytest.fixture
    def evaluator(self):
        config = LitellmConfig(model_name="gpt-5-nano")
        return BehaviorAdherenceEvaluator(model_config=config, num_workers=1)

    @pytest.fixture
    def sample_rollout_states(self):
        return [{"position": [0, 0]}, {"position": [1, 0]}]

    @pytest.fixture
    def sample_rollout_images(self):
        return [np.zeros((64, 64, 3), dtype=np.uint8)]

    def test_compute_with_images_success(self, mock_completion, mock_encode_images, evaluator, sample_rollout_states, sample_rollout_images):
        mock_completion.return_value = _mock_evaluation_response(True, "Visual confirms behavior")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": sample_rollout_images,
            }
        )

        assert result["adherence"] == 1.0
        assert "Visual confirms behavior" in result["reasoning"]
        mock_encode_images.assert_called_once()

    def test_compute_with_images_includes_images_in_content(self, mock_completion, mock_encode_images, evaluator, sample_rollout_states, sample_rollout_images):
        mock_encode_images.return_value = ["base64data"]
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.compute_instance_metric(
            {
                "behavior": behavior,
                "environment_description": "Test environment",
                "rollout_states": sample_rollout_states,
                "rollout_images": sample_rollout_images,
            }
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        assert isinstance(user_content, list)
        image_entries = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_entries) == 1
        assert "base64data" in image_entries[0]["image_url"]["url"]
