import json
import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation.policy_generator import PolicyGenerator, PolicyCode
from popgen.utils.parsing import clean_code


class TestPolicyCodeModel:
    def test_policy_code_creation(self):
        policy_code = PolicyCode(code="def agent0_policy(obs): pass\n\ndef agent1_policy(obs): pass")
        assert "agent0_policy" in policy_code.code
        assert "agent1_policy" in policy_code.code

    def test_policy_code_creation_with_reasoning(self):
        policy_code = PolicyCode(code="def agent0_policy(obs): pass", reasoning="Some reasoning")
        assert policy_code.reasoning == "Some reasoning"

    def test_policy_code_from_dict(self):
        data = {"code": "def agent0_policy(obs): return obs"}
        policy_code = PolicyCode.model_validate(data)
        assert "agent0_policy" in policy_code.code

    def test_policy_code_to_dict(self):
        policy_code = PolicyCode(code="code here", reasoning="reasoning here")
        data = policy_code.model_dump()
        assert data == {"code": "code here", "reasoning": "reasoning here"}


class TestPolicyGenerator:
    def test_init_custom_model(self):
        generator = PolicyGenerator(model="gpt-4-turbo")
        assert generator.model == "gpt-4-turbo"


class TestGenerate:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyGenerator(model="openai/gpt-4")

    @pytest.fixture
    def behavior(self):
        return '{"name": "Aggressive", "description": "Always moves toward target"}'

    def test_generate_success(self, mock_completion, generator, behavior):
        mock_response_0 = MagicMock()
        mock_response_0.choices = [MagicMock()]
        mock_response_0.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs):\n    return obs",
            "reasoning": "Move toward target"
        })

        mock_response_1 = MagicMock()
        mock_response_1.choices = [MagicMock()]
        mock_response_1.choices[0].message.content = json.dumps({
            "code": "def agent1_policy(obs):\n    return obs",
            "reasoning": "Coordinate with agent0"
        })

        mock_completion.side_effect = [mock_response_0, mock_response_1]

        policy_code = generator.generate(
            behavior=behavior,
            observation_format="obs is a tensor",
        )

        assert isinstance(policy_code, PolicyCode)
        assert "agent0_policy" in policy_code.code
        assert "agent1_policy" in policy_code.code
        assert mock_completion.call_count == 2


class TestGenerateStandalonePolicy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyGenerator(model="openai/gpt-4")

    @pytest.fixture
    def behavior(self):
        return '{"name": "Aggressive", "description": "Always moves toward target"}'

    def test_generate_standalone_policy_success(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs):\n    return obs",
            "reasoning": "Move aggressively"
        })
        mock_completion.return_value = mock_response

        policy = generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="obs is a tensor",
            agent_id=0,
        )

        assert isinstance(policy, PolicyCode)
        assert "agent0_policy" in policy.code
        mock_completion.assert_called_once()

    def test_generate_standalone_policy_includes_behavior_in_message(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="tensor format",
            agent_id=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Aggressive" in user_message

    def test_generate_standalone_policy_includes_observation_format(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="custom observation format here",
            agent_id=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        system_message = messages[0]["content"]
        assert "custom observation format here" in system_message

    def test_generate_standalone_policy_with_feedbacks(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs): return fixed",
            "reasoning": "Fixed the issue"
        })
        mock_completion.return_value = mock_response

        feedbacks = [
            ("def agent0_policy(obs): return broken", "Agent moved in wrong direction")
        ]

        generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="obs format",
            agent_id=0,
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
            "code": "def agent0_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="obs format",
            agent_id=0,
            feedbacks=[],
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2

    def test_generate_standalone_policy_no_refinement_with_none_feedbacks(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent0_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator._generate_standalone_policy(
            behavior=behavior,
            observation_format="obs format",
            agent_id=0,
            feedbacks=None,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2


class TestGenerateCoordinatingPolicy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy_generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyGenerator(model="openai/gpt-4")

    @pytest.fixture
    def agent0_behavior(self):
        return '{"name": "Leader", "description": "Takes initiative"}'

    @pytest.fixture
    def agent1_behavior(self):
        return '{"name": "Follower", "description": "Follows partner"}'

    def test_generate_coordinating_policy(self, mock_completion, generator, agent0_behavior, agent1_behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent1_policy(obs):\n    return coordinate(obs)",
            "reasoning": "Coordinate with leader"
        })
        mock_completion.return_value = mock_response

        agent0_code = "def agent0_policy(obs): return lead(obs)"

        policy = generator._generate_coordinating_policy(
            behavior=agent1_behavior,
            observation_format="obs format",
            agent_id=1,
            partner_behavior=agent0_behavior,
            partner_policy=agent0_code,
        )

        assert "agent1_policy" in policy.code
        mock_completion.assert_called_once()

    def test_coordinating_policy_includes_partner_info(self, mock_completion, generator, agent0_behavior, agent1_behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent1_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        agent0_code = "def agent0_policy(obs): return special_action(obs)"

        generator._generate_coordinating_policy(
            behavior=agent1_behavior,
            observation_format="obs format",
            agent_id=1,
            partner_behavior=agent0_behavior,
            partner_policy=agent0_code,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Leader" in user_message
        assert "special_action" in user_message
        assert "Follower" in user_message

    def test_coordinating_policy_uses_partner_prompt(self, mock_completion, generator, agent0_behavior, agent1_behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "code": "def agent1_policy(obs): pass",
            "reasoning": "reasoning"
        })
        mock_completion.return_value = mock_response

        generator._generate_coordinating_policy(
            behavior=agent1_behavior,
            observation_format="obs format",
            agent_id=1,
            partner_behavior=agent0_behavior,
            partner_policy="code",
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        system_message = messages[0]["content"]
        assert "coordinates with" in system_message.lower() or "partner" in system_message.lower()


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
