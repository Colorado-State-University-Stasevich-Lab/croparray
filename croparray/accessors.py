from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _BaseAccessor:
    parent: Any  # CropArray or TrackArray

    # Module path (relative to croparray package) used for delegation.
    # Each accessor overrides this, e.g. ".plot", ".measure", ".dataframe", etc.
    _delegate_module: Optional[str] = None

    @property
    def ds(self):
        # Both wrappers should expose .ds
        return self.parent.ds

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to a module associated with this accessor.

        This enables a lightweight workflow:
          - Put generic helper functions into the corresponding module
            (e.g. croparray/plot.py, croparray/measure.py, croparray/dataframe.py)
          - Access them as ca1.plot.<helper>(...), ca1.measure.<helper>(...), etc.
            without adding one-line wrapper methods each time.

        Notes
        -----
        - Delegation only triggers if normal attribute lookup fails.
        - Private names (starting with "_") are not delegated.
        - For dataset-aware functions that require `ds`, you can still provide
          explicit wrapper methods that pass `self.ds`.
        """
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        mod_path = getattr(self, "_delegate_module", None)
        if not mod_path:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        # Import lazily so edits + reloads are easier during development
        import importlib

        pkg = __package__  # e.g. "croparray"
        module = importlib.import_module(mod_path, package=pkg)

        try:
            return getattr(module, name)
        except AttributeError as e:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}") from e


# ----------------------------
# CropArray accessors
# ----------------------------

@dataclass
class CropArrayIO(_BaseAccessor):
    _delegate_module: Optional[str] = ".io"

    def open(self, *args, **kwargs):
        from .io import open_croparray
        return open_croparray(*args, **kwargs)

    def open_zarr(self, *args, **kwargs):
        from .io import open_croparray_zarr
        return open_croparray_zarr(*args, **kwargs)


@dataclass
class CropArrayBuild(_BaseAccessor):
    _delegate_module: Optional[str] = ".build"

    def create(self, *args, **kwargs):
        from .build import create_crop_array
        return create_crop_array(*args, **kwargs)


@dataclass
class CropArrayMeasure(_BaseAccessor):
    _delegate_module: Optional[str] = ".measure"

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
    # You can point this at a "front-door" ops module if you make one.
    # For now, delegate to the apply module (add more ops here later if desired).
    _delegate_module: Optional[str] = ".crop_ops.apply"

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
    _delegate_module: Optional[str] = ".plot"

    def montage(self, *args, **kwargs):
        from .plot import montage
        return montage(self.ds, *args, **kwargs)


@dataclass
class CropArrayView(_BaseAccessor):
    _delegate_module: Optional[str] = ".napari_view"

    def montage(self, *args, **kwargs):
        from .napari_view import view_montage
        return view_montage(*args, **kwargs)  # expects montage dataset/array


@dataclass
class CropArrayDF(_BaseAccessor):
    _delegate_module: Optional[str] = ".dataframe"

    def variables(self, var_names, *args, **kwargs):
        from .dataframe import variables_to_df
        return variables_to_df(self.ds, var_names, *args, **kwargs)


@dataclass
class CropArrayTrack(_BaseAccessor):
    _delegate_module: Optional[str] = ".tracking"

    def to_trackarray(self, *args, **kwargs):
        # calls existing functional tracker; returns TrackArray because you set as_object=True there
        from .tracking import to_track_array
        return to_track_array(self.ds, *args, **kwargs)


# ----------------------------
# TrackArray accessors
# ----------------------------

@dataclass
class TrackArrayPlot(_BaseAccessor):
    """
    Plotting utilities for TrackArray datasets.

    Wraps functions implemented in croparray/trackarray/plot.py.
    """

    # Optional: keep for future generic delegation if you want it later
    _delegate_module: Optional[str] = ".trackarray.plot"

    def plot_trackarray_crops(self, *args, **kwargs):
        from .trackarray.plot import plot_trackarray_crops
        return plot_trackarray_crops(self.ds, *args, **kwargs)

    def plot_track_signal_traces(self, *args, **kwargs):
        from .trackarray.plot import plot_track_signal_traces
        return plot_track_signal_traces(self.ds, *args, **kwargs)


@dataclass
class TrackArrayView(_BaseAccessor):
    # placeholder for napari viewers for trackarrays once you add them
    # Set this later when you create the module, e.g. ".napari_view"
    _delegate_module: Optional[str] = None


@dataclass
class TrackArrayDF(_BaseAccessor):
    _delegate_module: Optional[str] = ".dataframe"

    def variables(self, var_names, *args, **kwargs):
        from .dataframe import variables_to_df
        return variables_to_df(self.ds, var_names, *args, **kwargs)
