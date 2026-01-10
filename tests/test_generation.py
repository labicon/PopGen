import pytest
from unittest.mock import patch, MagicMock

from popgen.population.generation import (
    Behavior,
    BehaviorList,
    BehaviorGenerationStrategy,
    InContextStrategy,
    StratifiedStrategy,
)


class TestBehaviorModel:
    def test_behavior_creation(self):
        behavior = Behavior(name="Aggressive", description="Always attacks first")
        assert behavior.name == "Aggressive"
        assert behavior.description == "Always attacks first"

    def test_behavior_from_dict(self):
        data = {"name": "Defensive", "description": "Prioritizes safety"}
        behavior = Behavior.model_validate(data)
        assert behavior.name == "Defensive"
        assert behavior.description == "Prioritizes safety"

    def test_behavior_to_dict(self):
        behavior = Behavior(name="Balanced", description="Adapts to situation")
        data = behavior.model_dump()
        assert data == {"name": "Balanced", "description": "Adapts to situation"}

    def test_behavior_missing_field_raises(self):
        with pytest.raises(ValueError):
            Behavior(name="Incomplete")


class TestBehaviorListModel:
    def test_behavior_list_creation(self):
        behaviors = [
            Behavior(name="A", description="Desc A"),
            Behavior(name="B", description="Desc B"),
        ]
        behavior_list = BehaviorList(behaviors=behaviors)
        assert len(behavior_list.behaviors) == 2
        assert behavior_list.behaviors[0].name == "A"

    def test_behavior_list_from_dict(self):
        data = {
            "behaviors": [
                {"name": "X", "description": "Desc X"},
                {"name": "Y", "description": "Desc Y"},
            ]
        }
        behavior_list = BehaviorList.model_validate(data)
        assert len(behavior_list.behaviors) == 2

    def test_empty_behavior_list(self):
        behavior_list = BehaviorList(behaviors=[])
        assert len(behavior_list.behaviors) == 0


class TestBehaviorGenerationStrategy:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BehaviorGenerationStrategy()

    def test_behavior_list_to_string_empty(self):
        class ConcreteStrategy(BehaviorGenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy()
        result = str(BehaviorList(behaviors=[]))
        assert result == ""

    def test_behavior_list_to_string_with_behaviors(self):
        class ConcreteStrategy(BehaviorGenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy()
        behaviors = [
            Behavior(name="Alpha", description="First behavior"),
            Behavior(name="Beta", description="Second behavior"),
        ]
        result = str(BehaviorList(behaviors=behaviors))
        assert "Alpha: First behavior" in result
        assert "Beta: Second behavior" in result

    def test_default_model(self):
        class ConcreteStrategy(BehaviorGenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy()
        assert strategy.model == "openai/gpt-5"

    def test_custom_model(self):
        class ConcreteStrategy(BehaviorGenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy(model="claude-3-opus-20240229")
        assert strategy.model == "claude-3-opus-20240229"


def _mock_behavior_response(name: str, description: str) -> MagicMock:
    """Helper to create a mock LLM response with JSON content for a single Behavior."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = Behavior(
        name=name, description=description
    ).model_dump_json()
    return mock_response


def _mock_behavior_list_response(behaviors: list[Behavior]) -> MagicMock:
    """Helper to create a mock LLM response with JSON content for a BehaviorList."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = BehaviorList(
        behaviors=behaviors
    ).model_dump_json()
    return mock_response


def _mock_text_response(content: str) -> MagicMock:
    """Helper to create a mock LLM response with plain text content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestInContextStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.behavior.in_context.litellm.completion") as mock:
            yield mock

    def test_init_default_model(self):
        strategy = InContextStrategy()
        assert strategy.model == "openai/gpt-5"

    def test_init_custom_model(self):
        strategy = InContextStrategy(model="gpt-4-turbo")
        assert strategy.model == "gpt-4-turbo"

    def test_generate_single_success(self, mock_completion):
        mock_completion.return_value = _mock_behavior_response("Test", "Test description")

        strategy = InContextStrategy()
        result = strategy.generate_single("Test environment")

        assert result.name == "Test"
        assert result.description == "Test description"
        assert len(strategy.behavior_list) == 1
        mock_completion.assert_called_once()

    def test_generate_multiple_behaviors(self, mock_completion):
        mock_completion.side_effect = [
            _mock_behavior_response(f"Behavior{i}", f"Description {i}")
            for i in range(3)
        ]

        strategy = InContextStrategy()
        results = strategy.generate("Test environment", num_behaviors=3)

        assert len(results) == 3
        assert mock_completion.call_count == 3
        for i, result in enumerate(results):
            assert result.name == f"Behavior{i}"

    def test_generate_default_num_behaviors(self, mock_completion):
        mock_completion.return_value = _mock_behavior_response("Test", "Test")

        strategy = InContextStrategy()
        results = strategy.generate("Test environment")

        assert len(results) == 5
        assert mock_completion.call_count == 5

    def test_generate_single_retries_on_failure(self, mock_completion):
        mock_completion.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            _mock_behavior_response("Success", "After retry"),
        ]

        strategy = InContextStrategy()
        result = strategy.generate_single("Test environment")

        assert result.name == "Success"
        assert mock_completion.call_count == 3

    def test_generate_single_raises_after_max_retries(self, mock_completion):
        mock_completion.side_effect = Exception("Persistent failure")

        strategy = InContextStrategy()
        with pytest.raises(Exception, match="Persistent failure"):
            strategy.generate_single("Test environment")

        assert mock_completion.call_count == 5

    def test_previous_behaviors_included_in_prompt(self, mock_completion):
        mock_completion.return_value = _mock_behavior_response("New", "New behavior")

        strategy = InContextStrategy()
        strategy.behavior_list = [
            Behavior(name="Existing", description="Already generated")
        ]

        strategy.generate_single("Test environment")

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "Existing: Already generated" in messages[0]["content"]


class TestStratifiedStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.behavior.simplestrat.litellm.completion") as mock:
            yield mock

    def test_init_default_model(self):
        strategy = StratifiedStrategy()
        assert strategy.model == "openai/gpt-5"

    def test_init_custom_model(self):
        strategy = StratifiedStrategy(model="gpt-4-turbo")
        assert strategy.model == "gpt-4-turbo"

    def test_generate_single_divider(self, mock_completion):
        mock_completion.return_value = _mock_text_response("Risk tolerance: low vs high")

        strategy = StratifiedStrategy()
        result = strategy._generate_single_divider("Test environment", [])

        assert result == "Risk tolerance: low vs high"
        mock_completion.assert_called_once()

    def test_generate_single_divider_with_existing(self, mock_completion):
        mock_completion.return_value = _mock_text_response("Cooperation: selfish vs altruistic")

        strategy = StratifiedStrategy()
        result = strategy._generate_single_divider(
            "Test environment",
            ["Risk tolerance: low vs high"]
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "Risk tolerance: low vs high" in messages[0]["content"]

    def test_generate_behaviors_from_dividers(self, mock_completion):
        behaviors = [
            Behavior(name=f"B{i}", description=f"Desc {i}")
            for i in range(4)
        ]
        mock_completion.return_value = _mock_behavior_list_response(behaviors)

        strategy = StratifiedStrategy()
        result = strategy._generate_behaviors_from_dividers(
            "Test environment",
            "Divider 1",
            "Divider 2"
        )

        assert len(result) == 4
        assert len(strategy.behavior_list) == 4
        call_args = mock_completion.call_args
        assert call_args.kwargs["response_format"] == BehaviorList

    def test_generate_full_flow(self, mock_completion):
        behaviors = [
            Behavior(name=f"B{i}", description=f"Desc {i}")
            for i in range(4)
        ]

        mock_completion.side_effect = [
            _mock_text_response("Divider 1"),
            _mock_text_response("Divider 2"),
            _mock_behavior_list_response(behaviors),
        ]

        strategy = StratifiedStrategy()
        result = strategy.generate("Test environment")

        assert len(result) == 4
        assert mock_completion.call_count == 3

    def test_list_to_string_empty(self):
        strategy = StratifiedStrategy()
        result = strategy._list_to_string([])
        assert result == "None"

    def test_list_to_string_with_items(self):
        strategy = StratifiedStrategy()
        result = strategy._list_to_string(["Item 1", "Item 2"])
        assert "- Item 1" in result
        assert "- Item 2" in result
