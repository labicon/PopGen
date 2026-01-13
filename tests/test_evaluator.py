import pytest
from unittest.mock import patch, MagicMock

from popgen.population import (
    BehaviorEvaluator, 
    EvaluationResult,
    Behavior
)

class TestBehaviorEvaluationModel:
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
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def evaluator(self):
        return BehaviorEvaluator(model="gpt-5-nano")

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


class TestEvaluateTrajectory:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def evaluator(self):
        return BehaviorEvaluator(model="gpt-5-nano")

    def test_evaluate_trajectory_with_behavior_object(self, mock_completion, evaluator):
        mock_completion.return_value = _mock_evaluation_response(True, "Followed correctly")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.evaluate_trajectory("trajectory data", behavior)

        assert result.evaluation is True
        assert result.reasoning == "Followed correctly"
        mock_completion.assert_called_once()

    def test_evaluate_trajectory_with_dict(self, mock_completion, evaluator):
        mock_completion.return_value = _mock_evaluation_response(False, "Did not follow")

        behavior = {"name": "Test", "description": "Test behavior"}
        result = evaluator.evaluate_trajectory("trajectory data", behavior)

        assert result.evaluation is False
        assert result.reasoning == "Did not follow"

    def test_evaluate_trajectory_includes_behavior_in_prompt(self, mock_completion, evaluator):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Aggressive", description="Always attacks first")
        evaluator.evaluate_trajectory("some trajectory", behavior)

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Aggressive" in user_content
        assert "Always attacks first" in user_content
        assert "some trajectory" in user_content

    def test_evaluate_trajectory_retries_on_failure(self, mock_completion, evaluator):
        mock_completion.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            _mock_evaluation_response(True, "Success after retry"),
        ]

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_trajectory("trajectory", behavior)

        assert result.evaluation is True
        assert mock_completion.call_count == 3

    def test_evaluate_trajectory_returns_failure_after_max_retries(self, mock_completion, evaluator):
        mock_completion.side_effect = Exception("Persistent failure")

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_trajectory("trajectory", behavior)

        assert result.evaluation is False
        assert "Evaluation failed" in result.reasoning
        assert mock_completion.call_count == 5

    def test_evaluate_trajectory_uses_correct_response_format(self, mock_completion, evaluator):
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_trajectory("trajectory", behavior)

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == EvaluationResult


class TestEvaluateTrajectoryAndImages:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.evaluator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def mock_encode_image(self):
        with patch("popgen.population.evaluator.encode_image") as mock:
            mock.return_value = "base64encodedimage"
            yield mock

    @pytest.fixture
    def mock_encode_gif(self):
        with patch("popgen.population.evaluator.encode_gif") as mock:
            mock.return_value = ["frame1", "frame2"]
            yield mock

    @pytest.fixture
    def evaluator(self):
        return BehaviorEvaluator(model="gpt-5-nano")

    def test_evaluate_with_images_success(self, mock_completion, mock_encode_image, evaluator):
        mock_completion.return_value = _mock_evaluation_response(True, "Visual confirms behavior")

        behavior = Behavior(name="Test", description="Test behavior")
        result = evaluator.evaluate_trajectory_and_images(
            "trajectory data",
            ["/path/to/image.png"],
            behavior
        )

        assert result.evaluation is True
        assert result.reasoning == "Visual confirms behavior"
        mock_encode_image.assert_called_once_with("/path/to/image.png")

    def test_evaluate_with_multiple_images(self, mock_completion, mock_encode_image, evaluator):
        mock_encode_image.return_value = "encoded"
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_trajectory_and_images(
            "trajectory",
            ["/path/1.png", "/path/2.png", "/path/3.png"],
            behavior
        )

        assert mock_encode_image.call_count == 3

    def test_evaluate_with_images_includes_images_in_content(self, mock_completion, mock_encode_image, evaluator):
        mock_encode_image.return_value = "base64data"
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_trajectory_and_images("trajectory", ["/path/img.png"], behavior)

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
        assert "base64data" in user_content[1]["image_url"]["url"]

    def test_evaluate_with_gif_uses_encode_gif(self, mock_completion, mock_encode_gif, evaluator):
        mock_encode_gif.return_value = ["frame1", "frame2", "frame3"]
        mock_completion.return_value = _mock_evaluation_response(True, "OK")

        behavior = Behavior(name="Test", description="Test")
        evaluator.evaluate_trajectory_and_images("trajectory", ["/path/rollout.gif"], behavior)

        mock_encode_gif.assert_called_once_with("/path/rollout.gif")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        image_entries = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_entries) == 3

    def test_evaluate_with_images_returns_failure_on_encode_error(self, mock_encode_image, evaluator):
        mock_encode_image.side_effect = Exception("Cannot read image")

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_trajectory_and_images(
            "trajectory",
            ["/bad/path.png"],
            behavior
        )

        assert result.evaluation is False
        assert "Failed to read images" in result.reasoning

    def test_evaluate_with_images_retries_on_completion_failure(self, mock_completion, mock_encode_image, evaluator):
        mock_encode_image.return_value = "encoded"
        mock_completion.side_effect = [
            Exception("API error"),
            _mock_evaluation_response(True, "Success after retry"),
        ]

        behavior = Behavior(name="Test", description="Test")
        result = evaluator.evaluate_trajectory_and_images("trajectory", ["/path/img.png"], behavior)

        assert result.evaluation is True
        assert mock_completion.call_count == 2
