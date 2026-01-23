from typing import Any

from pydantic import BaseModel

from .base import GenerationStrategy
from .in_context import InContextStrategy
from .simplestrat import StratifiedStrategy
from .verbalized_sampling import VerbalizedSamplingStrategy

STRATEGY_REGISTRY: dict[str, type[GenerationStrategy]] = {
    "in_context": InContextStrategy,
    "stratified": StratifiedStrategy,
    "verbalized_sampling": VerbalizedSamplingStrategy,
}


def load_strategy(
    name: str,
    model: str,
    output_type: type[BaseModel],
    **kwargs: Any,
) -> GenerationStrategy:
    """Load a generation strategy by name.

    Args:
        name: The name of the strategy to load.
        model: The LLM model to use.
        output_type: The Pydantic model type for generated items.
        **kwargs: Additional arguments passed to the strategy constructor.

    Returns:
        An instance of the requested strategy.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    strategy_cls = STRATEGY_REGISTRY[name]
    return strategy_cls(model=model, output_type=output_type, **kwargs)


__all__ = [
    "GenerationStrategy",
    "InContextStrategy",
    "StratifiedStrategy",
    "VerbalizedSamplingStrategy",
    "load_strategy",
    "STRATEGY_REGISTRY",
]
