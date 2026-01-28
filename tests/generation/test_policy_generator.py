import json
import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation.policy_generator import PolicyCodeGenerator
from popgen.population.types import CodeSnippet
from popgen.utils.parsing import clean_code


class TestCodeSnippetModel:
    def test_code_snippet_creation(self):
        snippet = CodeSnippet(code="def policy(obs): pass")
        assert "policy" in snippet.code

    def test_code_snippet_creation_with_reasoning(self):
        snippet = CodeSnippet(code="def policy(obs): pass", reasoning="Some reasoning")
        assert snippet.reasoning == "Some reasoning"

    def test_code_snippet_from_dict(self):
        data = {"code": "def policy(obs): return obs"}
        snippet = CodeSnippet.model_validate(data)
        assert "def policy" in snippet.code

    def test_code_snippet_to_dict(self):
        snippet = CodeSnippet(code="code here", reasoning="reasoning here")
        data = snippet.model_dump()
        assert data == {"code": "code here", "reasoning": "reasoning here"}


class TestPolicyCodeGenerator:
    def test_init_custom_model(self):
        generator = PolicyCodeGenerator(model="gpt-4-turbo")
        assert generator.model == "gpt-4-turbo"


class TestGenerate:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyCodeGenerator(model="openai/gpt-4")

    @pytest.fixture
    def behavior(self):
        return '{"name": "Aggressive", "description": "Always moves toward target"}'

    def test_generate_success(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs):\n    return obs",
            "reasoning": "Move toward target"
        })

        mock_completion.return_value = mock_response

        policy_code = generator.generate(
            behavior=behavior,
            observation_format="obs is a tensor",
            action_format="action is a tensor",
        )

        assert isinstance(policy_code, CodeSnippet)
        assert "def policy" in policy_code.code
        assert mock_completion.call_count == 1


class TestGenerateStandalonePolicy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyCodeGenerator(model="openai/gpt-4")

    @pytest.fixture
    def behavior(self):
        return '{"name": "Aggressive", "description": "Always moves toward target"}'

    def test_generate_standalone_policy_success(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs):\n    return obs",
            "reasoning": "Move aggressively"
        })
        mock_completion.return_value = mock_response

        policy = generator.generate(
            behavior=behavior,
            observation_format="obs is a tensor",
            action_format="action is a tensor",
        )

        assert isinstance(policy, CodeSnippet)
        assert "def policy" in policy.code
        mock_completion.assert_called_once()

    def test_generate_standalone_policy_includes_behavior_in_message(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            observation_format="tensor format",
            action_format="action format",
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Aggressive" in user_message

    def test_generate_standalone_policy_includes_observation_format(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            observation_format="custom observation format here",
            action_format="action format",
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "custom observation format here" in user_message

    def test_generate_standalone_policy_with_feedbacks(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs): return fixed",
            "reasoning": "Fixed the issue"
        })
        mock_completion.return_value = mock_response

        feedbacks = [
            ("def policy(obs): return broken", "Agent moved in wrong direction")
        ]

        generator.generate(
            behavior=behavior,
            observation_format="obs format",
            action_format="action format",
            feedbacks=feedbacks,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        refinement_message = messages[2]["content"]
        assert "broken" in refinement_message
        assert "wrong direction" in refinement_message

    def test_generate_standalone_policy_no_refinement_with_empty_feedbacks(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            observation_format="obs format",
            action_format="action format",
            feedbacks=[],
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2

    def test_generate_standalone_policy_no_refinement_with_none_feedbacks(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator.generate(
            behavior=behavior,
            observation_format="obs format",
            action_format="action format",
            feedbacks=None,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2


class TestCleanCode:
    def test_clean_code_removes_python_markdown(self):
        code = "```python\ndef func(): pass\n```"
        result = clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_removes_generic_markdown(self):
        code = "```\ndef func(): pass\n```"
        result = clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_handles_no_markdown(self):
        code = "def func(): pass"
        result = clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_strips_whitespace(self):
        code = "  \n def func(): pass \n  "
        result = clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_handles_only_trailing_markdown(self):
        code = "def func(): pass\n```"
        result = clean_code(code)
        assert result == "def func(): pass"
