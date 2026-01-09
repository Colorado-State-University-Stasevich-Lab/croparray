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
        return f"TrackArray(tracks={self.ds.dims.get('track_id', '?')}, t={self.ds.dims.get('t', '?')})"

    @property
    def track_ids(self):
        return self.ds.coords["track_id"].values

    def sel_track(self, track_id: Union[int, list[int]]):
        """Return a TrackArray for one or more track IDs."""
        return TrackArray(self.ds.sel(track_id=track_id))

    def to_xarray(self) -> xr.Dataset:
        """Return the underlying xarray.Dataset."""
        return self.ds

    def _repr_html_(self) -> str:
        # Jupyter/IPython rich display
        try:
            return self.ds._repr_html_()
        except Exception:
            # Fall back to plain repr if something odd happens
            return f"<pre>{repr(self.ds)}</pre>"

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying xarray.Dataset.
        This makes ta.sel(...), ta.transpose(...), etc. work naturally.
        """
        return getattr(self.ds, name)

    def __getitem__(self, key):
        """Delegate ta['int'] etc. to the dataset."""
        return self.ds[key]
    
