"""Serialization adapters that convert external data into SampleSchema."""

from .base import BaseAdapter
from .mosei_pickle import MoseiPickleAdapter

__all__ = ["BaseAdapter", "MoseiPickleAdapter"]
