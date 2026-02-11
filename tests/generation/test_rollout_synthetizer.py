import json

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation.rollout_synthetizer import (
    RolloutSynthetizer,
    RolloutSynthetizerResult,
    BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
    SUMMARY_ANALYSIS_SYSTEM_PROMPT,
)


class TestRolloutSynthetizerResult:
    def test_creation_with_all_fields(self):
        result = RolloutSynthetizerResult(
            name="Aggressive",
            description="Moves toward opponents",
            reasoning="Agent consistently approaches enemies",
        )
        assert result.name == "Aggressive"
        assert result.description == "Moves toward opponents"
        assert result.reasoning == "Agent consistently approaches enemies"

    def test_str_method(self):
        result = RolloutSynthetizerResult(
            name="Defensive",
            description="Avoids conflict",
            reasoning="Agent stays away from danger",
        )
        expected = "Behavior: Defensive\nDescription: Avoids conflict\nReasoning: Agent stays away from danger"
        assert str(result) == expected

    def test_from_dict(self):
        data = {
            "name": "Explorer",
            "description": "Explores the map",
            "reasoning": "Agent visits new areas",
        }
        result = RolloutSynthetizerResult.model_validate(data)
        assert result.name == "Explorer"
        assert result.description == "Explores the map"
        assert result.reasoning == "Agent visits new areas"

    def test_to_dict(self):
        result = RolloutSynthetizerResult(
            name="Collector",
            description="Gathers resources",
            reasoning="Agent prioritizes pickups",
        )
        data = result.model_dump()
        assert data == {
            "name": "Collector",
            "description": "Gathers resources",
            "reasoning": "Agent prioritizes pickups",
        }

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError):
            RolloutSynthetizerResult(name="Test", description="Test")

    def test_from_json_string(self):
        json_str = json.dumps({
            "name": "Cautious",
            "description": "Careful movement",
            "reasoning": "Avoids risks",
        })
        result = RolloutSynthetizerResult.model_validate_json(json_str)
        assert result.name == "Cautious"


class TestRolloutSynthetizerInit:
    def test_init_with_model(self):
        analyzer = RolloutSynthetizer(model="gpt-4")
        assert analyzer.model == "gpt-4"

    def test_init_with_custom_model(self):
        analyzer = RolloutSynthetizer(model="openai/gpt-4-turbo")
        assert analyzer.model == "openai/gpt-4-turbo"


class TestAnalyzeSingleRollout:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.rollout_synthetizer.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def analyzer(self):
        return RolloutSynthetizer(model="openai/gpt-4")

    def test_analyze_single_rollout_success(self, mock_completion, analyzer):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Aggressive",
            "description": "Attacks enemies",
            "reasoning": "Agent moves toward opponents",
        })
        mock_completion.return_value = mock_response

        result = analyzer._analyze_single_rollout(
            env_description="A 2D grid world",
            rollout_states=[{"x": 0, "y": 0}, {"x": 1, "y": 1}],
            rollout_images=[],
            rollout_index=0,
        )

        assert isinstance(result, RolloutSynthetizerResult)
        assert result.name == "Aggressive"
        mock_completion.assert_called_once()

    def test_analyze_single_rollout_uses_correct_system_prompt(self, mock_completion, analyzer):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._analyze_single_rollout(
            env_description="A 2D grid world",
            rollout_states=[{"x": 0}],
            rollout_images=[],
            rollout_index=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == BEHAVIOR_ANALYSIS_SYSTEM_PROMPT

    def test_analyze_single_rollout_includes_env_description(self, mock_completion, analyzer):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._analyze_single_rollout(
            env_description="Grid environment",
            rollout_states=[{"x": 0}],
            rollout_images=[],
            rollout_index=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # User content is a list of dicts for multimodal
        first_text = user_content[0]["text"]
        assert "Grid environment" in first_text

    def test_analyze_single_rollout_with_images(self, mock_completion, analyzer):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        # Create a simple test image (2x2 RGB)
        test_image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)

        analyzer._analyze_single_rollout(
            env_description="Grid world",
            rollout_states=[{"x": 0}],
            rollout_images=[test_image],
            rollout_index=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Check that image content is included
        image_items = [item for item in user_content if item.get("type") == "image_url"]
        assert len(image_items) == 1
        assert "data:image/png;base64," in image_items[0]["image_url"]["url"]

    def test_analyze_single_rollout_error_handling(self, mock_completion, analyzer):
        mock_completion.side_effect = Exception("API Error")

        result = analyzer._analyze_single_rollout(
            env_description="Grid world",
            rollout_states=[{"x": 0}],
            rollout_images=[],
            rollout_index=0,
        )

        assert result.name == "Error"
        assert result.description == "Analysis failed."
        assert "API Error" in result.reasoning

    def test_analyze_single_rollout_uses_correct_model(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer = RolloutSynthetizer(model="custom-model")
        analyzer._analyze_single_rollout(
            env_description="Grid world",
            rollout_states=[{"x": 0}],
            rollout_images=[],
            rollout_index=0,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "custom-model"

    def test_analyze_single_rollout_passes_response_format(self, mock_completion, analyzer):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._analyze_single_rollout(
            env_description="Grid world",
            rollout_states=[{"x": 0}],
            rollout_images=[],
            rollout_index=0,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == RolloutSynthetizerResult


class TestSummarizeAnalyses:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.rollout_synthetizer.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def analyzer(self):
        return RolloutSynthetizer(model="openai/gpt-4")

    @pytest.fixture
    def sample_analyses(self):
        return [
            RolloutSynthetizerResult(name="Aggressive1", description="Attacks", reasoning="Reason1"),
            RolloutSynthetizerResult(name="Aggressive2", description="Fights", reasoning="Reason2"),
        ]

    def test_summarize_analyses_success(self, mock_completion, analyzer, sample_analyses):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Aggressive",
            "description": "Consistently attacks",
            "reasoning": "Both rollouts show aggressive behavior",
        })
        mock_completion.return_value = mock_response

        result = analyzer._summarize_analyses(sample_analyses, env_description="Grid world")

        assert isinstance(result, RolloutSynthetizerResult)
        assert result.name == "Aggressive"
        mock_completion.assert_called_once()

    def test_summarize_analyses_uses_correct_system_prompt(self, mock_completion, analyzer, sample_analyses):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._summarize_analyses(sample_analyses, env_description="Grid world")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SUMMARY_ANALYSIS_SYSTEM_PROMPT

    def test_summarize_analyses_includes_all_analyses(self, mock_completion, analyzer, sample_analyses):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._summarize_analyses(sample_analyses, env_description="Grid world")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Aggressive1" in user_message
        assert "Aggressive2" in user_message
        assert "Attacks" in user_message
        assert "Fights" in user_message

    def test_summarize_analyses_error_handling(self, mock_completion, analyzer, sample_analyses):
        mock_completion.side_effect = Exception("API Error")

        result = analyzer._summarize_analyses(sample_analyses, env_description="Grid world")

        assert result.name == "Error"
        assert result.description == "Summary failed."
        assert "API Error" in result.reasoning
        # Error should include the individual analyses
        assert "Aggressive1" in result.reasoning

    def test_summarize_analyses_includes_env_description(self, mock_completion, analyzer, sample_analyses):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "name": "Test",
            "description": "Test",
            "reasoning": "Test",
        })
        mock_completion.return_value = mock_response

        analyzer._summarize_analyses(sample_analyses, env_description="A complex maze environment")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "A complex maze environment" in user_message


class TestAnalyzeRollouts:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.rollout_synthetizer.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def analyzer(self):
        return RolloutSynthetizer(model="openai/gpt-4")

    def test_analyze_rollouts_calls_analyze_and_summarize(self, mock_completion, analyzer):
        # First two calls are for individual rollouts, third is for summary
        mock_responses = []
        for i in range(3):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({
                "name": f"Behavior{i}",
                "description": f"Description{i}",
                "reasoning": f"Reasoning{i}",
            })
            mock_responses.append(resp)
        mock_completion.side_effect = mock_responses

        # rollouts_states is a list of rollouts, where each rollout is a list of state dicts
        result = analyzer.analyze_rollouts(
            env_description="Grid world",
            rollouts_states=[[{"x": 0}], [{"x": 1}]],
            rollouts_images=None,
        )

        assert isinstance(result, RolloutSynthetizerResult)
        # 2 individual analyses + 1 summary = 3 calls
        assert mock_completion.call_count == 3

    def test_analyze_rollouts_with_images(self, mock_completion, analyzer):
        mock_responses = []
        for i in range(3):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({
                "name": f"Behavior{i}",
                "description": f"Description{i}",
                "reasoning": f"Reasoning{i}",
            })
            mock_responses.append(resp)
        mock_completion.side_effect = mock_responses

        test_image = np.zeros((2, 2, 3), dtype=np.uint8)

        result = analyzer.analyze_rollouts(
            env_description="Grid world",
            rollouts_states=[[{"x": 0}], [{"x": 1}]],
            rollouts_images=[[test_image], [test_image]],
        )

        assert isinstance(result, RolloutSynthetizerResult)
        assert mock_completion.call_count == 3

    def test_analyze_rollouts_none_images_creates_empty_lists(self, mock_completion, analyzer):
        mock_responses = []
        for i in range(2):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({
                "name": f"Behavior{i}",
                "description": f"Description{i}",
                "reasoning": f"Reasoning{i}",
            })
            mock_responses.append(resp)
        mock_completion.side_effect = mock_responses

        # Single rollout test - rollout is a list of state dicts
        analyzer.analyze_rollouts(
            env_description="Grid world",
            rollouts_states=[[{"x": 0}]],
            rollouts_images=None,
        )

        # Should still work without crashing
        assert mock_completion.call_count == 2

    def test_analyze_rollouts_passes_kwargs(self, mock_completion, analyzer):
        mock_responses = []
        for i in range(2):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({
                "name": f"Behavior{i}",
                "description": f"Description{i}",
                "reasoning": f"Reasoning{i}",
            })
            mock_responses.append(resp)
        mock_completion.side_effect = mock_responses

        analyzer.analyze_rollouts(
            env_description="Grid world",
            rollouts_states=[[{"x": 0}]],
            temperature=0.5,
            max_tokens=1000,
        )

        # Check that kwargs were passed to completion calls
        for call in mock_completion.call_args_list:
            assert call.kwargs.get("temperature") == 0.5
            assert call.kwargs.get("max_tokens") == 1000

    def test_analyze_rollouts_single_rollout(self, mock_completion, analyzer):
        mock_responses = []
        for i in range(2):  # 1 analysis + 1 summary
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps({
                "name": f"Behavior{i}",
                "description": f"Description{i}",
                "reasoning": f"Reasoning{i}",
            })
            mock_responses.append(resp)
        mock_completion.side_effect = mock_responses

        result = analyzer.analyze_rollouts(
            env_description="Grid world",
            rollouts_states=[[{"x": 0, "y": 1}]],
        )

        assert isinstance(result, RolloutSynthetizerResult)
        assert mock_completion.call_count == 2