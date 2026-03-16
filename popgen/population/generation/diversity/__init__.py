from typing import Any

from pydantic import BaseModel

from popgen.utils.config import LitellmConfig

from .base import GenerationStrategy
from .direct import DirectStrategy
from .simplestrat import StratifiedStrategy
from .verbalized_sampling import VerbalizedSamplingStrategy

STRATEGY_REGISTRY: dict[str, type[GenerationStrategy]] = {
    "direct": DirectStrategy,
    "stratified": StratifiedStrategy,
    "verbalized_sampling": VerbalizedSamplingStrategy,
}


def load_strategy(
    strategy: str,
    model_config: LitellmConfig,
    output_type: type[BaseModel],
    **kwargs: Any,
) -> GenerationStrategy:
    """Load a generation strategy by name.

    Args:
        strategy: The name of the strategy to load.
        model_config: The LLM model configuration to use.
        output_type: The Pydantic model type for generated items.
        **kwargs: Additional arguments passed to the strategy constructor.

    Returns:
        An instance of the requested strategy.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if strategy not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")

    strategy_cls = STRATEGY_REGISTRY[strategy]
    return strategy_cls(model_config=model_config, output_type=output_type, **kwargs)


__all__ = [
    "GenerationStrategy",
    "DirectStrategy",
    "StratifiedStrategy",
    "VerbalizedSamplingStrategy",
    "load_strategy",
    "STRATEGY_REGISTRY",
]
