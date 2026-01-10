from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _BaseAccessor:
    parent: Any  # CropArray or TrackArray

    @property
    def ds(self):
        # Both wrappers should expose .ds
        return self.parent.ds


# ----------------------------
# CropArray accessors
# ----------------------------

@dataclass
class CropArrayIO(_BaseAccessor):
    def open(self, *args, **kwargs):
        from .io import open_croparray
        return open_croparray(*args, **kwargs)

    def open_zarr(self, *args, **kwargs):
        from .io import open_croparray_zarr
        return open_croparray_zarr(*args, **kwargs)


@dataclass
class CropArrayBuild(_BaseAccessor):
    def create(self, *args, **kwargs):
        from .build import create_crop_array
        return create_crop_array(*args, **kwargs)


@dataclass
class CropArrayMeasure(_BaseAccessor):
    def best_z_proj(self, *args, **kwargs):
        from .measure import best_z_proj
        return best_z_proj(self.ds, *args, **kwargs)

    def signal(self, *args, **kwargs):
        from .measure import measure_signal
        return measure_signal(self.ds, *args, **kwargs)

    def signal_raw(self, *args, **kwargs):
        from .measure import measure_signal_raw
        return measure_signal_raw(self.ds, *args, **kwargs)


@dataclass
class CropArrayOps(_BaseAccessor):
    def apply(self, func, source="best_z", *args, **kwargs):
        """
        Apply a single-crop function across the crop array using xr.apply_ufunc.

        Parameters
        ----------
        func : callable
            Function that operates on a single crop (e.g. (x,y) array) and returns
            either a scalar or an image.
        source : str
            Name of the DataArray in the dataset to operate on (default: "best_z").

        Other args/kwargs are forwarded to apply_crop_op.
        """
        from .crop_ops.apply import apply_crop_op
        return apply_crop_op(self.ds, func, source=source, *args, **kwargs)


@dataclass
class CropArrayPlot(_BaseAccessor):
    def montage(self, *args, **kwargs):
        from .plot import montage
        return montage(self.ds, *args, **kwargs)


@dataclass
class CropArrayView(_BaseAccessor):
    def montage(self, *args, **kwargs):
        from .napari_view import view_montage
        return view_montage(*args, **kwargs)  # expects montage dataset/array


@dataclass
class CropArrayDF(_BaseAccessor):
    def variables(self, var_names, *args, **kwargs):
        from .dataframe import variables_to_df
        return variables_to_df(self.ds, var_names, *args, **kwargs)


@dataclass
class CropArrayTrack(_BaseAccessor):
    def to_trackarray(self, *args, **kwargs):
        # calls existing functional tracker; returns TrackArray because you set as_object=True there
        from .tracking import to_track_array
        return to_track_array(self.ds, *args, **kwargs)


# ----------------------------
# TrackArray accessors
# ----------------------------

@dataclass
class TrackArrayPlot(_BaseAccessor):
    # placeholder for trackarray plot utilities once you add them
    pass


@dataclass
class TrackArrayView(_BaseAccessor):
    # placeholder for napari viewers for trackarrays once you add them
    pass


@dataclass
class TrackArrayDF(_BaseAccessor):
    def variables(self, var_names, *args, **kwargs):
        from .dataframe import variables_to_df
        return variables_to_df(self.ds, var_names, *args, **kwargs)
