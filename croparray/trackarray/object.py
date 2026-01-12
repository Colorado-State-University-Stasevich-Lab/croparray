from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional

import xarray as xr


@dataclass
class TrackArray:
    """
    Lightweight object wrapper around a track-array xarray.Dataset.

    The underlying dataset is expected to have at least:
      - dimension: track_id
      - dimension: t
    and often:
      - dimension: fov (length 1 per track)
      - optional dims: z, y, x, ch
    """
    ds: xr.Dataset

    def __post_init__(self):
        if not isinstance(self.ds, xr.Dataset):
            raise TypeError("TrackArray expects an xarray.Dataset")
        if "track_id" not in self.ds.dims:
            raise ValueError("TrackArray dataset must have a 'track_id' dimension")

    def __repr__(self) -> str:
        return f"TrackArray(ds={repr(self.ds)})"

    @property
    def track_ids(self):
        return self.ds.coords["track_id"].values

    def sel_track(self, track_id: Union[int, list[int]]):
        """Return a TrackArray for one or more track IDs."""
        return TrackArray(self.ds.sel(track_id=track_id))

    def to_xarray(self) -> xr.Dataset:
        """Return the underlying xarray.Dataset."""
        return self.ds
