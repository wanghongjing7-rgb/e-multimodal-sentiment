"""Serialization adapters that convert external data into SampleSchema."""

from .base import BaseAdapter
from .attachment2_aligned import Attachment2AlignedAdapter
from .mosei_pickle import MoseiPickleAdapter

__all__ = ["Attachment2AlignedAdapter", "BaseAdapter", "MoseiPickleAdapter"]
