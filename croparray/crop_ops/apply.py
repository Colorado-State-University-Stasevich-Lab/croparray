from typing import Any, Callable, Dict, Optional, Sequence, Union
import xarray as xr


def apply_crop_op(
    ds: xr.Dataset,
    func: Callable,
    source: Union[str, xr.DataArray] = "best_z",
    *,
    channels: Optional[Union[int, Sequence[int]]] = None,
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
    Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
    optionally iterating over channels, and optionally computing a per-crop scalar
    summary (e.g., summed signal).

    Conceptually, this function executes the “known-good” pattern:

        xr.apply_ufunc(
            func,
            da.sel(ch=<one channel>),
            input_core_dims=[["x", "y"]],
            output_core_dims=[["x", "y"]],
            kwargs=...,
            vectorize=True,
        )

    and does so for every crop (and for each requested channel, if present).

    Parameters
    ----------
    ds
        The dataset containing crop data variables (e.g., a stack of crops across time/track/crop_id).
    func
        A function to apply to each crop. It must accept an array-like input matching
        `input_core_dims` (typically a 2D image) and return an array-like output matching
        `output_core_dims`. Additional keyword arguments may be provided via `func_kwargs`.

        Typical examples: spot detection / QC, filtering, background subtraction, etc.
    source
        Either the name of a DataArray in `ds` (default: `"best_z"`) or an explicit
        `xr.DataArray` to process.
    channels
        Channel selection behavior when `channel_dim` exists in the source DataArray:

        - `None` (default): process **all** channels present in `da.coords[channel_dim]`.
        - `int`: process only that channel (e.g., `channels=0`).
        - `Sequence[int]`: process only those channels (e.g., `channels=[0]` or `[0, 1]`).

        If the source DataArray does **not** contain `channel_dim`, this argument is ignored
        and the operation is applied once (i.e., “single-channel” mode).
    channel_dim
        Name of the channel dimension in the source DataArray (default: `"ch"`).
    input_core_dims
        Core dimensions consumed by `func` (default: `("x", "y")`). These are passed to
        `xr.apply_ufunc(..., input_core_dims=[...])`.
    output_core_dims
        Core dimensions produced by `func` (default: `("x", "y")`). These are passed to
        `xr.apply_ufunc(..., output_core_dims=[...])`.
    out_name
        Output variable name template. If channel iteration is used, `{ch}` is formatted
        with the channel index (e.g., `"ch0_spots"`). If no channel dimension exists,
        `{ch}` is formatted as `"NA"` by default.
    func_kwargs
        Optional keyword arguments forwarded to `func`. If `None`, treated as `{}`.
    compute_sum_xy
        If True, compute an additional summary signal per crop by summing `out` over
        `sum_dims` and store it as a second output variable.
    sum_name
        Name template for the summed-signal output, formatted with `{out}` set to the
        generated `out_name` (default: `"{out}_sig"`).
    sum_dims
        Dimensions to sum over when `compute_sum_xy=True` (default: `("x", "y")`).
        Use this to control whether you sum only spatial dims, or include others.
    add_to_ds
        If True (default), add outputs back into `ds` and return the updated dataset.
        If False, return a dict mapping output variable names to DataArrays.

    Returns
    -------
    xr.Dataset or Dict[str, xr.DataArray]
        - If `add_to_ds=True`: the input dataset `ds`, augmented with new variables.
        - If `add_to_ds=False`: a dict of created outputs (and optional summaries).

    Notes
    -----
    - This function relies on `vectorize=True` in `xr.apply_ufunc`, meaning `func` is
      applied independently across non-core dimensions (crop index, time, track_id, etc.).
    - Channel handling is done via `da.sel({channel_dim: ch})` per channel.
    - If you request channels that are not present in the DataArray coordinates,
      a clear `ValueError` is raised.

    Examples
    --------
    Process all channels in `best_z`:

        ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                           out_name="ch{ch}_spots",
                           func_kwargs={"minmass": 150, "size": 3})

    Process only channel 0:

        ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                           channels=[0],
                           out_name="ch{ch}_spots",
                           func_kwargs={"minmass": 150, "size": 3})

    Process channels 0 and 1 and also compute summed signal per crop:

        ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                           channels=[0, 1],
                           out_name="ch{ch}_spots",
                           func_kwargs={"minmass": 150, "size": 3},
                           compute_sum_xy=True,
                           sum_name="{out}_sig")
    """
    func_kwargs = {} if func_kwargs is None else dict(func_kwargs)

    da = ds[source] if isinstance(source, str) else source

    # Normalize channel selection.
    def _normalize_channels(
        da_: xr.DataArray,
        channels_: Optional[Union[int, Sequence[int]]],
        channel_dim_: str,
    ) -> Sequence[Optional[int]]:
        # No channel dimension => run once.
        if channel_dim_ not in da_.dims:
            return [None]

        available = list(da_.coords[channel_dim_].values)

        # Default: all channels.
        if channels_ is None:
            # Coerce to python ints where reasonable (for clean formatting later).
            return [int(c) for c in available]

        # Accept a single int.
        if isinstance(channels_, (int,)):
            requested = [int(channels_)]
        else:
            requested = [int(c) for c in channels_]

        missing = [c for c in requested if c not in list(map(int, available))]
        if missing:
            raise ValueError(
                f"Requested channels {missing} not found in {channel_dim_} coords. "
                f"Available: {list(map(int, available))}"
            )

        return requested

    channel_list = _normalize_channels(da, channels, channel_dim)

    created: Dict[str, xr.DataArray] = {}

    for ch in channel_list:
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
