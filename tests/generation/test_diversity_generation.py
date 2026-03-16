import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, create_model

from popgen.population.generation.diversity import (
    GenerationStrategy,
    DirectStrategy,
    StratifiedStrategy,
)
from popgen.utils.config import LitellmConfig


class SampleItem(BaseModel):
    """A sample item for generation testing."""
    name: str
    description: str


_SampleItemList = create_model("SampleItemList", items=(list[SampleItem], ...))


def _mock_item_response(name: str, description: str) -> MagicMock:
    """Helper to create a mock LLM response with JSON content for a single SampleItem."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = SampleItem(
        name=name, description=description
    ).model_dump_json()
    return mock_response


def _mock_item_list_response(items: list[SampleItem]) -> MagicMock:
    """Helper to create a mock LLM response with JSON content for a list of SampleItems."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = _SampleItemList(items=items).model_dump_json()
    return mock_response


def _mock_text_response(content: str) -> MagicMock:
    """Helper to create a mock LLM response with plain text content."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _mock_property_list_response(properties: list[str]) -> MagicMock:
    """Helper to create a mock LLM response with PropertyList JSON content."""
    from popgen.population.generation.diversity.simplestrat import PropertyList
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = PropertyList(properties=properties).model_dump_json()
    return mock_response


class TestGenerationStrategy:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            config = LitellmConfig(model_name="openai/gpt-4")
            GenerationStrategy(model_config=config, output_type=SampleItem)

    def test_list_to_string_empty(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = ConcreteStrategy(model_config=config, output_type=SampleItem)
        result = strategy._list_to_string([])
        assert result == "None"

    def test_list_to_string_with_items(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = ConcreteStrategy(model_config=config, output_type=SampleItem)
        items = [
            SampleItem(name="Alpha", description="First item"),
            SampleItem(name="Beta", description="Second item"),
        ]
        result = strategy._list_to_string(items)
        assert "Alpha" in result
        assert "Beta" in result

    def test_custom_model(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        config = LitellmConfig(model_name="claude-3-opus-20240229")
        strategy = ConcreteStrategy(model_config=config, output_type=SampleItem)
        assert strategy.model_config.model_name == "claude-3-opus-20240229"

    def test_output_type_stored(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = ConcreteStrategy(model_config=config, output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generated_items_initialized_empty(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = ConcreteStrategy(model_config=config, output_type=SampleItem)
        assert strategy.generated_items == []


class TestDirectStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.diversity.direct.litellm.completion") as mock:
            yield mock

    def test_init_custom_model(self):
        config = LitellmConfig(model_name="gpt-4-turbo")
        strategy = DirectStrategy(model_config=config, output_type=SampleItem)
        assert strategy.model_config.model_name == "gpt-4-turbo"

    def test_init_output_type(self):
        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(model_config=config, output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generate_single_call(self, mock_completion):
        items = [SampleItem(name=f"Item{i}", description=f"Desc {i}") for i in range(4)]
        mock_completion.return_value = _mock_item_list_response(items)

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate("Test environment", num_items=4)

        assert len(result) == 4
        assert mock_completion.call_count == 1
        for i, item in enumerate(result):
            assert item.name == f"Item{i}"

    def test_generate_default_num_items(self, mock_completion):
        items = [SampleItem(name=f"Item{i}", description=f"Desc {i}") for i in range(4)]
        mock_completion.return_value = _mock_item_list_response(items)

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate("Test environment")

        assert len(result) == 4
        assert mock_completion.call_count == 1

    def test_multiturn_sequential(self, mock_completion):
        """samples_per_turn=1, independent=False: 4 sequential calls with context."""
        mock_completion.side_effect = [
            _mock_item_list_response([SampleItem(name=f"Item{i}", description=f"Desc {i}")])
            for i in range(4)
        ]

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(
            model_config=config, output_type=SampleItem,
            samples_per_turn=1, independent_multiturns=False,
        )
        result = strategy.generate("Test environment", num_items=4)

        assert len(result) == 4
        assert mock_completion.call_count == 4
        # Second call should include context from first item
        second_call_messages = mock_completion.call_args_list[1].kwargs["messages"]
        assert "Item0" in second_call_messages[1]["content"]

    def test_multiturn_independent(self, mock_completion):
        """samples_per_turn=1, independent=True: 4 parallel calls without context."""
        mock_completion.side_effect = [
            _mock_item_list_response([SampleItem(name=f"Item{i}", description=f"Desc {i}")])
            for i in range(4)
        ]

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(
            model_config=config, output_type=SampleItem,
            samples_per_turn=1, independent_multiturns=True,
        )
        result = strategy.generate("Test environment", num_items=4)

        assert len(result) == 4
        assert mock_completion.call_count == 4
        # All calls should have the same base prompt (no context from previous items)
        for call in mock_completion.call_args_list:
            user_msg = call.kwargs["messages"][1]["content"]
            assert user_msg == "Test environment"

    def test_generated_items_tracked(self, mock_completion):
        items = [SampleItem(name="A", description="a"), SampleItem(name="B", description="b")]
        mock_completion.return_value = _mock_item_list_response(items)

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = DirectStrategy(model_config=config, output_type=SampleItem)
        strategy.generate("Test", num_items=2)

        assert len(strategy.generated_items) == 2


class TestStratifiedStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.diversity.simplestrat.litellm.completion") as mock:
            yield mock

    def test_init_custom_model(self):
        config = LitellmConfig(model_name="gpt-4-turbo")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        assert strategy.model_config.model_name == "gpt-4-turbo"

    def test_init_output_type(self):
        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generate_properties(self, mock_completion):
        mock_completion.return_value = _mock_property_list_response(["Property 1", "Property 2"])

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate_properties("Test environment", num_properties=2, candidate_rate=1.0)

        assert result == ["Property 1", "Property 2"]
        mock_completion.assert_called_once()

    def test_generate_properties_prompt_contains_request(self, mock_completion):
        mock_completion.return_value = _mock_property_list_response(["Property 1"])

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        strategy.generate_properties("My specific request", num_properties=1, candidate_rate=1.0)

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "My specific request" in messages[1]["content"]

    def test_generate_item_from_properties(self, mock_completion):
        mock_completion.return_value = _mock_item_response("TestItem", "Test description")

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate_item_from_properties(
            "Test environment",
            ["Property 1", "Property 2"]
        )

        assert result.name == "TestItem"
        assert result.description == "Test description"
        mock_completion.assert_called_once()

    @patch("popgen.population.generation.diversity.simplestrat.random.random")
    def test_generate_full_flow(self, mock_random, mock_completion):
        # Mock random to deterministically select properties
        mock_random.return_value = 0.3  # < 0.5, so all properties selected

        mock_completion.side_effect = [
            _mock_property_list_response(["Property 1", "Property 2"]),
            _mock_item_response("I0", "Desc 0"),
            _mock_item_response("I1", "Desc 1"),
            _mock_item_response("I2", "Desc 2"),
            _mock_item_response("I3", "Desc 3"),
        ]

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate("Test environment")

        assert len(result) == 4
        # 1 call for properties + 4 calls for items
        assert mock_completion.call_count == 5

    @patch("popgen.population.generation.diversity.simplestrat.random.random")
    def test_generate_with_custom_num_items(self, mock_random, mock_completion):
        # Mock random to deterministically select properties
        mock_random.return_value = 0.3  # < 0.5, so all properties selected

        mock_completion.side_effect = [
            _mock_property_list_response(["Property 1", "Property 2", "Property 3"]),
            _mock_item_response("I0", "Desc 0"),
            _mock_item_response("I1", "Desc 1"),
            _mock_item_response("I2", "Desc 2"),
            _mock_item_response("I3", "Desc 3"),
            _mock_item_response("I4", "Desc 4"),
            _mock_item_response("I5", "Desc 5"),
        ]

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy.generate("Test environment", num_items=6)

        assert len(result) == 6

    def test_list_to_string_empty(self):
        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy._list_to_string([])
        assert result == "None"

    def test_list_to_string_with_items(self):
        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = StratifiedStrategy(model_config=config, output_type=SampleItem)
        result = strategy._list_to_string(["Item 1", "Item 2"])
        assert "- Item 1" in result
        assert "- Item 2" in result
