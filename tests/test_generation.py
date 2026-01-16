import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from popgen.population.generation.diversity import (
    GenerationStrategy,
    InContextStrategy,
    StratifiedStrategy,
)


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


class TestGenerationStrategy:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            GenerationStrategy(model="openai/gpt-4", output_type=SampleItem)

    def test_list_to_string_empty(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._list_to_string([])
        assert result == "None"

    def test_list_to_string_with_items(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy(model="openai/gpt-4", output_type=SampleItem)
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

        strategy = ConcreteStrategy(model="claude-3-opus-20240229", output_type=SampleItem)
        assert strategy.model == "claude-3-opus-20240229"

    def test_output_type_stored(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy(model="openai/gpt-4", output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generated_items_initialized_empty(self):
        class ConcreteStrategy(GenerationStrategy):
            def generate(self, environment_description, **kwargs):
                return []

        strategy = ConcreteStrategy(model="openai/gpt-4", output_type=SampleItem)
        assert strategy.generated_items == []


class TestInContextStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.diversity.in_context.litellm.completion") as mock:
            yield mock

    def test_init_custom_model(self):
        strategy = InContextStrategy(model="gpt-4-turbo", output_type=SampleItem)
        assert strategy.model == "gpt-4-turbo"

    def test_init_output_type(self):
        strategy = InContextStrategy(model="openai/gpt-4", output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generate_single_success(self, mock_completion):
        mock_completion.return_value = _mock_item_response("Test", "Test description")

        strategy = InContextStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._generate_single("Test environment")

        assert result.name == "Test"
        assert result.description == "Test description"
        assert len(strategy.generated_items) == 1
        mock_completion.assert_called_once()

    def test_generate_multiple_items(self, mock_completion):
        mock_completion.side_effect = [
            _mock_item_response(f"Item{i}", f"Description {i}")
            for i in range(3)
        ]

        strategy = InContextStrategy(model="openai/gpt-4", output_type=SampleItem)
        results = strategy.generate("Test environment", num_items=3)

        assert len(results) == 3
        assert mock_completion.call_count == 3
        for i, result in enumerate(results):
            assert result.name == f"Item{i}"

    def test_generate_default_num_items(self, mock_completion):
        mock_completion.return_value = _mock_item_response("Test", "Test")

        strategy = InContextStrategy(model="openai/gpt-4", output_type=SampleItem)
        results = strategy.generate("Test environment")

        assert len(results) == 4
        assert mock_completion.call_count == 4

    def test_previous_items_included_in_prompt(self, mock_completion):
        mock_completion.return_value = _mock_item_response("New", "New item")

        strategy = InContextStrategy(model="openai/gpt-4", output_type=SampleItem)
        existing_items = [SampleItem(name="Existing", description="Already generated")]

        strategy._generate_single("Test environment", previous_items=existing_items)

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "Existing" in messages[0]["content"]


class TestStratifiedStrategy:
    @pytest.fixture
    def mock_completion(self):
        with patch("popgen.population.generation.diversity.simplestrat.litellm.completion") as mock:
            yield mock

    def test_init_custom_model(self):
        strategy = StratifiedStrategy(model="gpt-4-turbo", output_type=SampleItem)
        assert strategy.model == "gpt-4-turbo"

    def test_init_output_type(self):
        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        assert strategy.output_type == SampleItem

    def test_generate_single_divider(self, mock_completion):
        mock_completion.return_value = _mock_text_response("Risk tolerance: low vs high")

        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._generate_single_divider("Test environment", [])

        assert result == "Risk tolerance: low vs high"
        mock_completion.assert_called_once()

    def test_generate_single_divider_with_existing(self, mock_completion):
        mock_completion.return_value = _mock_text_response("Cooperation: selfish vs altruistic")

        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._generate_single_divider(
            "Test environment",
            ["Risk tolerance: low vs high"]
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        assert "Risk tolerance: low vs high" in messages[0]["content"]

    def test_generate_items_from_dividers(self, mock_completion):
        items = [
            SampleItem(name=f"I{i}", description=f"Desc {i}")
            for i in range(4)
        ]
        mock_completion.return_value = _mock_item_list_response(items)

        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._generate_items_from_dividers(
            "Test environment",
            ["Divider 1", "Divider 2"],
            num_items=4
        )

        assert len(result) == 4
        assert len(strategy.generated_items) == 4

    def test_generate_full_flow(self, mock_completion):
        items = [
            SampleItem(name=f"I{i}", description=f"Desc {i}")
            for i in range(4)
        ]

        mock_completion.side_effect = [
            _mock_text_response("Divider 1"),
            _mock_text_response("Divider 2"),
            _mock_item_list_response(items),
        ]

        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy.generate("Test environment")

        assert len(result) == 4
        assert mock_completion.call_count == 3

    def test_generate_with_custom_num_items(self, mock_completion):
        items = [
            SampleItem(name=f"I{i}", description=f"Desc {i}")
            for i in range(6)
        ]

        mock_completion.side_effect = [
            _mock_text_response("Divider 1"),
            _mock_text_response("Divider 2"),
            _mock_item_list_response(items),
        ]

        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy.generate("Test environment", num_items=6)

        assert len(result) == 6

    def test_list_to_string_empty(self):
        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._list_to_string([])
        assert result == "None"

    def test_list_to_string_with_items(self):
        strategy = StratifiedStrategy(model="openai/gpt-4", output_type=SampleItem)
        result = strategy._list_to_string(["Item 1", "Item 2"])
        assert "- Item 1" in result
        assert "- Item 2" in result
