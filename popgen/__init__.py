import warnings

# Suppress Pydantic serialization warning from litellm's Message object
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic.main",
)
