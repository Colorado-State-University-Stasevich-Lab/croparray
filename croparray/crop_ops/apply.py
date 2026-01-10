import numpy as np
import xarray as xr
from typing import Callable, Sequence, Optional, Union, Dict, Any

def apply_crop_op(
    ds: xr.Dataset,
    func: Callable,
    source: Union[str, xr.DataArray] = "best_z",
    *,
    channels: Optional[Sequence[int]] = None,
    channel_dim: str = "ch",
    input_core_dims: Sequence[str] = ("x", "y"),
    output_core_dims: Sequence[str] = ("x", "y"),
    out_name: str = "ch{ch}_op",
    func_kwargs: Optional[Dict[str, Any]] = None,
    compute_sum_xy: bool = False,
    sum_name: str = "{out}_sig",
    sum_dims: Sequence[str] = ("x", "y"),
    add_to_ds: bool = True,
) -> Union[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    Apply a single-crop function (battle-tested via xr.apply_ufunc)
    across all crops, optionally looping over channels.

    This function intentionally mirrors the known-working pattern:

        xr.apply_ufunc(
            func,
            da.sel(ch=...),
            input_core_dims=[["x","y"]],
            output_core_dims=[["x","y"]],
            vectorize=True,
        )
    """

    func_kwargs = {} if func_kwargs is None else dict(func_kwargs)

    da = ds[source] if isinstance(source, str) else source

    # Determine channels
    if channel_dim in da.dims:
        if channels is None:
            channels = list(da.coords[channel_dim].values)
    else:
        channels = [None]

    created: Dict[str, xr.DataArray] = {}

    for ch in channels:
        da_in = da.sel({channel_dim: ch}) if ch is not None else da

        out = xr.apply_ufunc(
            func,
            da_in,
            input_core_dims=[list(input_core_dims)],
            output_core_dims=[list(output_core_dims)],
            kwargs=func_kwargs,
            vectorize=True,
        )

        ch_label = "NA" if ch is None else int(ch)
        out_var = out_name.format(ch=ch_label)
        out.name = out_var
        created[out_var] = out

        if compute_sum_xy:
            sig = out.sum(list(sum_dims))
            sig_name = sum_name.format(out=out_var)
            sig.name = sig_name
            created[sig_name] = sig

    if add_to_ds:
        for k, v in created.items():
            ds[k] = v
        return ds

    return created
