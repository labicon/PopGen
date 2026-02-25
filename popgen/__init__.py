import logging
import warnings

import litellm

# Suppress LiteLLM INFO logs
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Suppress Pydantic serialization warning from litellm's Message object
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic.main",
)
