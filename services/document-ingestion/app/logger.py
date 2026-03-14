# Copied from pgvector source as-is (portable — no platform-specific deps).
# Removed OpenTelemetry hooks that were present in some variants.

import structlog
from .config import Settings

config = Settings()


def add_service_name(_, __, event_dict):
    """
    Add the configured service name to a structlog event dictionary.
    
    Parameters:
        event_dict (dict): The event dictionary being built by structlog.
    
    Returns:
        dict: The same event dictionary with a `service_name` key set to the application's display name.
    """
    event_dict["service_name"] = config.APP_DISPLAY_NAME
    return event_dict


# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        add_service_name,
        structlog.processors.JSONRenderer(),
    ]
)

# Module-level logger used everywhere in the package
logger = structlog.get_logger()
