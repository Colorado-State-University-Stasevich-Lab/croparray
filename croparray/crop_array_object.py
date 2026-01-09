
from dataclasses import dataclass

import xarray as xr

from .tracking import to_track_array as _to_track_array


@dataclass
class CropArray:
    """
    Object wrapper around the underlying xarray.Dataset produced by the original
    crop-array builder. Provides method-style API: ca.best_z_proj(...), etc.
    """
    ds: xr.Dataset

    def __post_init__(self):
        """
        Attach namespaced accessors (io/build/measure/plot/view/df/track)
        so you can call e.g.:
            ca1.plot.montage(...)
            ca1.measure.best_z_proj(...)
            ca1.track.to_trackarray(...)
        """
        from .accessors import (
            CropArrayIO,
            CropArrayBuild,
            CropArrayMeasure,
            CropArrayPlot,
            CropArrayView,
            CropArrayDF,
            CropArrayTrack,
        )

        self.io = CropArrayIO(self)
        self.build = CropArrayBuild(self)
        self.measure = CropArrayMeasure(self)
        self.plot = CropArrayPlot(self)
        self.view = CropArrayView(self)
        self.df = CropArrayDF(self)
        self.track = CropArrayTrack(self)

    
    def __getitem__(self, key: str):
        return self.ds[key]

    def __getattr__(self, name: str):
        return getattr(self.ds, name)

    def to_xarray(self):
        """Return the underlying xarray.Dataset."""
        return self.ds

    def best_z_proj(self, ref_ch: int = 0, disk_r: int = 1, roll_n: int = 1):
        """
        Compute the best-z projection of crop intensities and add/overwrite `ds['zc']`.

        Parameters
        ----------
        ref_ch : int or None, default 0
            Reference channel used to choose the best z-plane. If None, best-z is
            computed separately for each channel.
        disk_r : int, default 1
            Radius (pixels) of the centered XY disk used to score each z-plane.
        roll_n : int, default 1
            Number of z-slices used in a rolling-z max projection (min_periods=1).

        Returns
        -------
        xarray.DataArray
            Best-z projection with dims like (fov, n, f, y, x, ch). Also augments
            the underlying dataset by adding/overwriting `ds['zc']`.
        """
        from .measure import best_z_proj
        out = best_z_proj(self.ds, ref_ch=ref_ch, disk_r=disk_r, roll_n=roll_n)

        self.ds["best_z"] = out
        return out
    
    def measure_signal(
        self,
        ref_ch=None,
        disk_r: int = 1,
        disk_bg=None,
        roll_n: int = 1,
        **kwargs
    ):
        """
        Measure background-subtracted intensity signals for crops.

        Parameters
        ----------
        ref_ch : int or None, default None
            Channel used to choose best z for measurements. None uses all channels.
        disk_r : int, default 1
            Radius (pixels) for signal measurement disk.
        disk_bg : int or None, default None
            Radius (pixels) defining an outer ring (width 1 pixel) for background.
            If None, the function may default to `xy_pad` behavior.
        roll_n : int, default 1
            Rolling-z window used for z selection/projection.
        **kwargs
            Passed through to the underlying implementation.

        Returns
        -------
        CropArray
            Self, with `ds` augmented (e.g., adds `best_z`, `signal`, etc.).
        """
        from .measure import measure_signal
        ds2 = measure_signal(
            self.ds,
            ref_ch=ref_ch,
            disk_r=disk_r,
            disk_bg=disk_bg,
            roll_n=roll_n,
            **kwargs
        )

        if isinstance(ds2, xr.Dataset) and ds2 is not self.ds:
            self.ds = ds2
        return self
    
    def to_trackarray(
    self,
    channel_to_track: int = 0,
    min_track_length: int = 5,
    search_range: int = 10,
    memory: int = 1,
    ):
        """
        Track particles in this CropArray and return a TrackArray object.

        Notes
        -----
        This overwrites `self.ds['id']` to store track IDs (0 indicates untracked/filtered).
        The original `id` is preserved in `spot_id` the first time tracking is run.
        """
        return _to_track_array(
            self.ds,
            channel_to_track=channel_to_track,
            min_track_length=min_track_length,
            search_range=search_range,
            memory=memory,
        )
