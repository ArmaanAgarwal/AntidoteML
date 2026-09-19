"""Person 1's area: the model, the data, and how a model becomes a vector.

Everything in the frozen contract is re-exported here, so the rest of the repo
can use `from antidote.ml import ...` without caring which file it lives in.
"""

from antidote.ml.flat import get_flat, set_flat
from antidote.ml.model import make_model
from antidote.ml.train import local_train

__all__ = ["get_flat", "local_train", "make_model", "set_flat"]
