import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from popgen.population.generation.diversity import (
    GenerationStrategy,
    InContextStrategy,
    StratifiedStrategy,
)
from popgen.utils.config import LitellmConfig


class SampleItem(BaseModel):
    """A sample item for generation testing."""
    name: str
    description: str


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
    # Create dynamic list model
    from pydantic import create_model
    list_model = create_model("SampleItemList", items=(list[SampleItem], ...))
    mock_response.choices[0].message.content = list_model(items=items).model_dump_json()
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


class TestInContextStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.diversity.in_context.litellm.completion") as mock:
            yield mock

    def test_init_custom_model(self):
        config = LitellmConfig(model_name="gpt-4-turbo")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        assert strategy.model_config.model_name == "gpt-4-turbo"

    def test_init_output_type(self):
        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generate_single_success(self, mock_completion):
        mock_completion.return_value = _mock_item_response("Test", "Test description")

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        result = strategy._generate_single("Test environment")

        assert result.name == "Test"
        assert result.description == "Test description"
        # _generate_single doesn't add to generated_items; only generate() does
        assert len(strategy.generated_items) == 0
        mock_completion.assert_called_once()

    def test_generate_multiple_items(self, mock_completion):
        mock_completion.side_effect = [
            _mock_item_response(f"Item{i}", f"Description {i}")
            for i in range(3)
        ]

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        results = strategy.generate("Test environment", num_items=3)

        assert len(results) == 3
        assert mock_completion.call_count == 3
        for i, result in enumerate(results):
            assert result.name == f"Item{i}"

    def test_generate_default_num_items(self, mock_completion):
        mock_completion.return_value = _mock_item_response("Test", "Test")

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        results = strategy.generate("Test environment")

        assert len(results) == 4
        assert mock_completion.call_count == 4

    def test_previous_items_included_in_prompt(self, mock_completion):
        mock_completion.return_value = _mock_item_response("New", "New item")

        config = LitellmConfig(model_name="openai/gpt-4")
        strategy = InContextStrategy(model_config=config, output_type=SampleItem)
        existing_items = [SampleItem(name="Existing", description="Already generated")]

        strategy._generate_single("Test environment", previous_items=existing_items)

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        # Previous items are in user prompt (messages[1]), not system prompt (messages[0])
        assert "Existing" in messages[1]["content"]


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
