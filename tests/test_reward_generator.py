import json
import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation.reward_generator import RewardFnGenerator, Reward
from popgen.utils.parsing import clean_code


class TestRewardModel:
    def test_reward_creation(self):
        reward = Reward(code="def agent0_reward(obs, action, next_obs, done): return 0.0")
        assert reward.code == "def agent0_reward(obs, action, next_obs, done): return 0.0"
        assert reward.reasoning is None

    def test_reward_creation_with_reasoning(self):
        reward = Reward(
            code="def agent0_reward(obs, action, next_obs, done): return 1.0",
            reasoning="Simple constant reward"
        )
        assert reward.reasoning == "Simple constant reward"

    def test_reward_from_dict(self):
        data = {"code": "def agent0_reward(obs, action, next_obs, done): return obs[0]"}
        reward = Reward.model_validate(data)
        assert "agent0_reward" in reward.code

    def test_reward_to_dict(self):
        reward = Reward(code="code here", reasoning="some reasoning")
        data = reward.model_dump()
        assert data == {"code": "code here", "reasoning": "some reasoning"}

    def test_reward_to_dict_without_reasoning(self):
        reward = Reward(code="code here")
        data = reward.model_dump()
        assert data == {"code": "code here", "reasoning": None}

    def test_reward_missing_code_raises(self):
        with pytest.raises(ValueError):
            Reward(reasoning="reasoning without code")


class TestRewardFnGenerator:
    def test_init_custom_model(self):
        generator = RewardFnGenerator(model="gpt-4-turbo")
        assert generator.model == "gpt-4-turbo"


class TestGenerate:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.reward_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return RewardFnGenerator(model="openai/gpt-4")

    @pytest.fixture
    def behavior(self):
        return '{"name": "Aggressive", "description": "Always moves toward target"}'

    @pytest.fixture
    def codebase_context(self):
        return "obs is a tensor with shape (4,) containing [x, y, vx, vy]"

    def test_generate_success(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done):\n    return 1.0",
            "reasoning": "Reward for moving toward target"
        })
        mock_completion.return_value = mock_response

        reward = generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
        )

        assert isinstance(reward, Reward)
        assert "agent0_reward" in reward.code
        mock_completion.assert_called_once()

    def test_generate_includes_behavior_in_message(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): return 0",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Aggressive" in user_message

    def test_generate_includes_codebase_context_in_system_prompt(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): return 0",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        system_message = messages[0]["content"]
        assert "obs is a tensor with shape (4,)" in system_message

    def test_generate_with_feedbacks(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): return fixed",
            "reasoning": "Fixed the issue"
        })
        mock_completion.return_value = mock_response

        feedbacks = [
            ("def agent0_reward(obs, action, next_obs, done): return broken", "IndexError: tensor index out of range")
        ]

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
            feedbacks=feedbacks,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        refinement_message = messages[2]["content"]
        assert "broken" in refinement_message
        assert "IndexError" in refinement_message

    def test_generate_with_multiple_feedbacks(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): return fixed",
            "reasoning": "Fixed the issues"
        })
        mock_completion.return_value = mock_response

        feedbacks = [
            ("code1", "error1"),
            ("code2", "error2"),
        ]

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
            feedbacks=feedbacks,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        refinement_message = messages[2]["content"]
        assert "code1" in refinement_message
        assert "error1" in refinement_message
        assert "code2" in refinement_message
        assert "error2" in refinement_message

    def test_generate_no_refinement_with_empty_feedbacks(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
            feedbacks=[],
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2

    def test_generate_no_refinement_with_none_feedbacks(self, mock_completion, generator, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
            feedbacks=None,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2

    def test_generate_uses_correct_model(self, mock_completion, behavior, codebase_context):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_reward(obs, action, next_obs, done): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator = RewardFnGenerator(model="custom-model")
        generator.generate(
            behavior=behavior,
            codebase_context=codebase_context,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "custom-model"


class TestCleanCode:
    def test_clean_code_removes_python_markdown(self):
        code = "```python\ndef agent0_reward(obs, action, next_obs, done): pass\n```"
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_removes_generic_markdown(self):
        code = "```\ndef agent0_reward(obs, action, next_obs, done): pass\n```"
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_handles_no_markdown(self):
        code = "def agent0_reward(obs, action, next_obs, done): pass"
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_strips_whitespace(self):
        code = "  \n def agent0_reward(obs, action, next_obs, done): pass \n  "
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_handles_only_trailing_markdown(self):
        code = "def agent0_reward(obs, action, next_obs, done): pass\n```"
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_handles_only_leading_python_markdown(self):
        code = "```python\ndef agent0_reward(obs, action, next_obs, done): pass"
        result = clean_code(code)
        assert result == "def agent0_reward(obs, action, next_obs, done): pass"

    def test_clean_code_handles_multiline_code(self):
        code = """```python
def agent0_reward(obs, action, next_obs, done):
    distance = obs[0]
    return -distance
```"""
        result = clean_code(code)
        expected = """def agent0_reward(obs, action, next_obs, done):
    distance = obs[0]
    return -distance"""
        assert result == expected
