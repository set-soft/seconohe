from dataclasses import dataclass
import os


@dataclass
class SeconoheConfig:
    """
    Configuration for the seconohe library.

    Attributes:
        NODES_NAME: The name of the nodes
    """
    NODES_NAME: str = "Unknown"
    JS_PATH: str = os.path.join(os.path.dirname(__file__), "js")
