
from dataclasses import dataclass

import xarray as xr

from . import crop_array_tools as ca

@dataclass
class CropArray:
    """
    Object wrapper around the underlying xarray.Dataset produced by the original
    crop-array builder. Provides method-style API: ca.best_z_proj(...), etc.
    """
    ds: xr.Dataset

    def __getitem__(self, key: str):
        return self.ds[key]

    def __getattr__(self, name: str):
        return getattr(self.ds, name)

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
        out = ca.best_z_proj(self.ds, ref_ch=ref_ch, disk_r=disk_r, roll_n=roll_n)
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
        ds2 = ca.measure_signal(
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