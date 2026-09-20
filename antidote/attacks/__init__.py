"""Attack simulations used to evaluate AntidoteML's defenses."""

from .poison import maybe_poison
from .tamper import maybe_tamper
from .trigger import apply_trigger

__all__ = ["apply_trigger", "maybe_poison", "maybe_tamper"]
