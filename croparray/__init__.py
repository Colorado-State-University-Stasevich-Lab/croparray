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
]
