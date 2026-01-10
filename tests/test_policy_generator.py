import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation import PolicyGenerator, Policy, Behavior


class TestPolicyModel:
    def test_policy_creation(self):
        policy = Policy(code="def agent0_policy(obs): pass", agent_id=0)
        assert policy.code == "def agent0_policy(obs): pass"
        assert policy.agent_id == 0

    def test_policy_from_dict(self):
        data = {"code": "def agent1_policy(obs): return obs", "agent_id": 1}
        policy = Policy.model_validate(data)
        assert policy.agent_id == 1
        assert "agent1_policy" in policy.code

    def test_policy_to_dict(self):
        policy = Policy(code="code here", agent_id=0)
        data = policy.model_dump()
        assert data == {"code": "code here", "agent_id": 0}

    def test_policy_missing_field_raises(self):
        with pytest.raises(ValueError):
            Policy(code="some code")


class TestPolicyGenerator:
    def test_init_default_model(self):
        generator = PolicyGenerator()
        assert generator.model == "openai/gpt-5"

    def test_init_custom_model(self):
        generator = PolicyGenerator(model="gpt-4-turbo")
        assert generator.model == "gpt-4-turbo"


class TestGenerateStandalonePolicy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy.generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyGenerator()

    @pytest.fixture
    def behavior(self):
        return Behavior(name="Aggressive", description="Always moves toward target")

    def test_generate_policy_success(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent0_policy(obs):\n    return obs"
        mock_completion.return_value = mock_response

        policy = generator.generate_policy(
            behavior=behavior,
            observation_format="obs is a tensor",
            agent_id=0,
        )

        assert isinstance(policy, Policy)
        assert policy.agent_id == 0
        assert "agent0_policy" in policy.code
        mock_completion.assert_called_once()

    def test_generate_policy_for_agent1(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent1_policy(obs):\n    return obs"
        mock_completion.return_value = mock_response

        policy = generator.generate_policy(
            behavior=behavior,
            observation_format="obs is a tensor",
            agent_id=1,
        )

        assert policy.agent_id == 1
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "agent1" in messages[0]["content"]

    def test_generate_policy_includes_behavior_in_message(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent0_policy(obs): pass"
        mock_completion.return_value = mock_response

        generator.generate_policy(
            behavior=behavior,
            observation_format="tensor format",
            agent_id=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Aggressive" in user_message

    def test_generate_policy_includes_observation_format(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent0_policy(obs): pass"
        mock_completion.return_value = mock_response

        generator.generate_policy(
            behavior=behavior,
            observation_format="custom observation format here",
            agent_id=0,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        system_message = messages[0]["content"]
        assert "custom observation format here" in system_message

    def test_generate_policy_with_refinement(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent0_policy(obs): return fixed"
        mock_completion.return_value = mock_response

        previous = "def agent0_policy(obs): return broken"
        error = "Agent moved in wrong direction"

        policy = generator.generate_policy(
            behavior=behavior,
            observation_format="obs format",
            agent_id=0,
            previous_policy=previous,
            error_message=error,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        refinement_message = messages[2]["content"]
        assert "broken" in refinement_message
        assert "wrong direction" in refinement_message

    def test_generate_policy_refinement_not_added_without_both_params(self, mock_completion, generator, behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent0_policy(obs): pass"
        mock_completion.return_value = mock_response

        generator.generate_policy(
            behavior=behavior,
            observation_format="obs format",
            agent_id=0,
            previous_policy="some code",
            error_message=None,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2


class TestGenerateCoordinatingPolicy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.policy.generator.litellm.completion") as mock:
            yield mock

    @pytest.fixture
    def generator(self):
        return PolicyGenerator()

    @pytest.fixture
    def agent0_behavior(self):
        return Behavior(name="Leader", description="Takes initiative")

    @pytest.fixture
    def agent1_behavior(self):
        return Behavior(name="Follower", description="Follows partner")

    def test_generate_coordinating_policy(self, mock_completion, generator, agent0_behavior, agent1_behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent1_policy(obs):\n    return coordinate(obs)"
        mock_completion.return_value = mock_response

        agent0_code = "def agent0_policy(obs): return lead(obs)"

        policy = generator.generate_policy(
            behavior=agent1_behavior,
            observation_format="obs format",
            agent_id=1,
            partner_behavior=agent0_behavior,
            partner_policy=agent0_code,
        )

        assert policy.agent_id == 1
        assert "agent1_policy" in policy.code
        mock_completion.assert_called_once()

    def test_coordinating_policy_includes_partner_info(self, mock_completion, generator, agent0_behavior, agent1_behavior):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "def agent1_policy(obs): pass"
        mock_completion.return_value = mock_response

        agent0_code = "def agent0_policy(obs): return special_action(obs)"

        generator.generate_policy(
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
        mock_response.choices[0].message.content = "def agent1_policy(obs): pass"
        mock_completion.return_value = mock_response

        generator.generate_policy(
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
    @pytest.fixture
    def generator(self):
        return PolicyGenerator()

    def test_clean_code_removes_python_markdown(self, generator):
        code = "```python\ndef func(): pass\n```"
        result = generator._clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_removes_generic_markdown(self, generator):
        code = "```\ndef func(): pass\n```"
        result = generator._clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_handles_no_markdown(self, generator):
        code = "def func(): pass"
        result = generator._clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_strips_whitespace(self, generator):
        code = "  \n def func(): pass \n  "
        result = generator._clean_code(code)
        assert result == "def func(): pass"

    def test_clean_code_handles_only_trailing_markdown(self, generator):
        code = "def func(): pass\n```"
        result = generator._clean_code(code)
        assert result == "def func(): pass"
