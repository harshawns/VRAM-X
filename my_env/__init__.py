"""VRAM Exchange: Multi-agent GPU negotiation environment."""

from .client import VRAMExchange
from .models import VRAMAction, VRAMObservation

__all__ = [
    "VRAMAction",
    "VRAMObservation",
    "VRAMExchange",
]


def __getattr__(name: str):
    """Defer client import so non-client utilities can be imported independently."""
    if name == "MyEnv":
        from .client import MyEnv

        return MyEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
