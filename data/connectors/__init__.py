"""Dataset connectors package.

Importing this package registers all available loaders in the registry
(data.connectors.base) so `get_loader(...)` works out of the box.
"""

from . import huggingface_loader, kaggle_loader, movielens_loader  # noqa: F401
from .base import BaseDatasetLoader, LoadedData, available_loaders, get_loader

__all__ = ["BaseDatasetLoader", "LoadedData", "available_loaders", "get_loader"]
