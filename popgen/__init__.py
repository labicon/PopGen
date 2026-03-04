import logging

import litellm
from popgen.utils.warnings import configure_warning_filters

# Suppress LiteLLM INFO logs
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

configure_warning_filters()
