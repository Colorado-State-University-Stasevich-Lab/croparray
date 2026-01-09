# -*- coding: utf-8 -*-
"""
croparray

Created on Nov 26 2021
@author: Tim Stasevich
"""

__version__ = "0.0.9"

# Object-oriented API
from .crop_array_object import CropArray


# Functional / tools API (explicitly attached)
from . import crop_array_tools

# Short alias for functional tools (preferred over top-level forwarding)
tools = crop_array_tools

# Convenience re-exports (top-level shortcuts)
from .crop_array_tools import (
    create_crop_array,
    open_croparray,
    open_croparray_zarr,
)

__all__ = [
    "CropArray",
    "create_crop_array",
    "open_croparray",
    "open_croparray_zarr",
    "crop_array_tools",
    "tools",
]

# ---------------------------------------------------------------------
# Backward compatibility: allow croparray.<function> to resolve to
# croparray.crop_array_tools.<function>, with a warning.
# ---------------------------------------------------------------------
import warnings as _warnings

def __getattr__(name: str):
    if hasattr(crop_array_tools, name):
        _warnings.warn(
            f"croparray.{name} is provided for backward compatibility and "
            f"may be deprecated in the future.\n"
            f"Use croparray.tools.{name}(...) or the CropArray method "
            f"(e.g., ca1.{name}(...)) when available.",
            category=FutureWarning,
            stacklevel=2,
        )
        return getattr(crop_array_tools, name)
    raise AttributeError(f"module 'croparray' has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(dir(crop_array_tools)))
