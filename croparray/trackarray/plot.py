from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt



def plot_trackarray_crops(
    data,
    fov=0,
    track_id=1,
    t=(0, 10, 3),
    rolling=1,
    quantile_range=(0.02, 0.99),
    rgb_channels=(0, 1, 2),
    ch=None,
    suppress_labels=True,
    show_suptitle=True,
):
    """
    Plot track-centered image crops across time and channels using xarray.plot.imshow.

    For each track:
      - Optionally shows an RGB composite row (if >=3 channels and ch is None)
      - Shows grayscale rows for each channel (or only `ch` if provided)

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing a 'best_z' DataArray with dims like:
        (track_id, fov, t, y, x, ch) or (track_id, t, y, x, ch).
    fov : int, default 0
        Field of view to select, if applicable.
    track_id : int or sequence[int], default 1
        Track(s) to plot. One grid per track.
    t : tuple[int, int, int], default (0, 10, 3)
        (start, stop, step) along time axis.
    rolling : int, default 1
        Rolling mean window over time. If 1, no smoothing.
    quantile_range : tuple[float, float], default (0.02, 0.99)
        Quantiles used for per-channel normalization (positive pixels only).
    rgb_channels : tuple, default (0, 1, 2)
        Channels to use for RGB composite when available.
        Duplicates are allowed (e.g. (1, 1, 2)).
    ch : int or None, default None
        If provided, plot only this channel in grayscale.
    suppress_labels : bool, default True
        If True, removes per-crop titles, axis labels, and ticks.
    show_suptitle : bool, default True
        If True, adds a readable figure-level title per track.

    Returns
    -------
    dict[int, xr.DataArray]
        Mapping from track_id -> normalized DataArray used for plotting
        with dims (t, y, x, ch). If `ch` is provided, ch size = 1.
    """
    import xarray as xr

    if "best_z" not in data:
        raise KeyError("Dataset must contain a 'best_z' DataArray.")

    def _decorate_facetgrid(g, suptitle=None):
        """Optionally suppress per-panel labels and add a figure-level title."""
        if suppress_labels:
            # Remove facet titles (e.g. t=0, t=3, ...)
            try:
                g.set_titles("")
            except Exception:
                try:
                    g.set_titles(template="")
                except Exception:
                    pass

            # Remove axis labels
            try:
                g.set_xlabels("")
                g.set_ylabels("")
            except Exception:
                pass

        if show_suptitle and suptitle and hasattr(g, "fig") and g.fig is not None:
            try:
                g.fig.suptitle(suptitle, y=1.02)
            except Exception:
                pass

        return g

    track_ids = (
        list(track_id)
        if isinstance(track_id, (list, tuple, np.ndarray))
        else [track_id]
    )

    eps = 1e-6
    results = {}

    for tid in track_ids:
        bz = data["best_z"].sel(track_id=tid)

        # --- FOV selection ---
        if "fov" in bz.dims:
            bz = bz.sel(fov=fov)
        elif "fov" in data.coords:
            try:
                bz = bz.where(data["fov"] == fov, drop=True)
            except Exception:
                pass

        # --- Time slicing ---
        start, stop, step = t
        bz = bz.isel(t=slice(start, stop, step))

        # --- Rolling average ---
        if rolling and rolling > 1:
            bz = bz.rolling(t=rolling, center=True, min_periods=1).mean()

        # --- Single-channel mode ---
        if ch is not None:
            try:
                bz1 = bz.sel(ch=ch)
            except Exception:
                bz1 = bz.isel(ch=int(ch))

            bz1 = bz1.expand_dims("ch").assign_coords(ch=[ch])

            da_pos = bz1.isel(ch=0).where(lambda x: x > 0)
            q0 = da_pos.quantile(quantile_range[0])
            q1 = da_pos.quantile(quantile_range[1])

            normed = (
                ((bz1.isel(ch=0) - q0) / (q1 - q0 + eps))
                .clip(0, 1)
                .expand_dims(ch=[ch])
            )

            g = normed.isel(ch=0).plot.imshow(
                col="t",
                cmap="gray",
                xticks=[] if suppress_labels else None,
                yticks=[] if suppress_labels else None,
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=True,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )
            _decorate_facetgrid(g, suptitle=f"track_id={tid} | ch={ch}")

            results[int(tid)] = normed
            continue

        # --- Per-channel normalization ---
        ch_normed = []
        for ch_val in bz["ch"].values:
            da_pos = bz.sel(ch=ch_val).where(lambda x: x > 0)
            q0 = da_pos.quantile(quantile_range[0])
            q1 = da_pos.quantile(quantile_range[1])
            ch_normed.append(
                ((bz.sel(ch=ch_val) - q0) / (q1 - q0 + eps)).clip(0, 1)
            )

        normed = xr.concat(ch_normed, dim="ch").assign_coords(ch=bz["ch"].values)

        # --- RGB composite ---
        if normed.sizes.get("ch", 0) >= 3:
            try:
                rgb_da = normed.sel(ch=list(rgb_channels))
            except Exception:
                rgb_da = normed.isel(ch=slice(0, 3))

            if rgb_da.sizes.get("ch", 0) == 3:
                g = rgb_da.plot.imshow(
                    col="t",
                    rgb="ch",
                    xticks=[] if suppress_labels else None,
                    yticks=[] if suppress_labels else None,
                    aspect=1,
                    size=5,
                    vmin=0,
                    vmax=1,
                    add_labels=not suppress_labels,
                    add_colorbar=False,
                )
                _decorate_facetgrid(g, suptitle=f"track_id={tid} (RGB)")

        # --- Grayscale rows ---
        for ch_val in normed["ch"].values:
            g = normed.sel(ch=ch_val).plot.imshow(
                col="t",
                cmap="gray",
                xticks=[] if suppress_labels else None,
                yticks=[] if suppress_labels else None,
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=True,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )
            _decorate_facetgrid(g, suptitle=f"track_id={tid}, ch={ch_val}")

        results[int(tid)] = normed

    return results



def plot_track_signal_traces(
    ta_dataset,
    track_ids,
    var: str = "signal",
    rgb=(1, 1, 1),
    colors=("#00f670", "#f67000", "#7000f6"),
    markers=("o", "s", "D"),
    marker_size=6,
    scatter_size=25,
    markevery=5,
    figsize=(7, 2.8),
    ylim=None,
    xlim=None,
    col_wrap=3,
    y2=None,
    y2lim=None,
    y2_label=None,
    legend_loc="upper right",
    show_legend=True,
):
    """
    Plot per-track traces for a chosen variable (default: 'signal') in a subplot grid.
    Optionally place one channel on a secondary (right) y-axis.

    Parameters
    ----------
    ta_dataset : xarray.Dataset
        TrackArray dataset containing `var` with dims including (track_id, t) and usually (ch).
    track_ids : list[int]
        Track IDs to plot.
    var : str, default "signal"
        Variable name to plot (e.g. "signal", "signal_raw", "ID", "MEAN_INTENSITY").
    rgb : tuple[int,int,int] or None, default (1,1,1)
        Channel inclusion mask for left axis (unless a channel is assigned to y2).
        If None, plot all channels.
    colors, markers, marker_size, scatter_size, markevery, figsize, ylim, xlim, col_wrap, y2, y2lim, y2_label,
    legend_loc, show_legend : see previous docstring.
    """
    from ..dataframe import variables_to_df
    import math
    import seaborn as sns
    import matplotlib.pyplot as plt

    if var not in ta_dataset:
        raise KeyError(f"Dataset does not contain variable '{var}'")

    df = variables_to_df(ta_dataset, [var])

    sns.set_style("whitegrid")
    sns.set(font_scale=1.1)

    n = len(track_ids)
    ncols = col_wrap
    nrows = math.ceil(n / col_wrap)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False,
    )

    # Ensure markers covers all channels
    if len(markers) < len(colors):
        markers = list(markers) + ["o"] * (len(colors) - len(markers))

    def _rgb_on(ch: int) -> bool:
        if rgb is None:
            return True
        if ch < len(rgb):
            return bool(rgb[ch])
        return False

    for idx, track_id in enumerate(track_ids):
        row, col = divmod(idx, col_wrap)
        ax = axes[row][col]
        ax2 = ax.twinx() if y2 is not None else None

        for ch in range(len(colors)):
            if not (_rgb_on(ch) or (y2 == ch)):
                continue

            color = colors[ch]
            marker = markers[ch]

            subset = df[(df["track_id"] == track_id) & (df["ch"] == ch)]
            if subset.empty:
                continue

            target_ax = ax2 if (y2 is not None and ch == y2) else ax

            sns.lineplot(
                data=subset,
                x="t",
                y=var,
                ax=target_ax,
                color=color,
                label=f"ch {ch}",
                lw=2,
                dashes=False,
                legend=False,
                marker=marker,
                markersize=marker_size,
                markevery=markevery,
            )

            mean_df = subset.groupby("t")[var].mean().reset_index()
            sns.scatterplot(
                data=mean_df,
                x="t",
                y=var,
                ax=target_ax,
                color=color,
                s=scatter_size,
                legend=False,
            )

        ax.set_title(f"Track {int(track_id)}")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel(f"{var} (a.u.)")
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

        if ax2 is not None:
            right_color = colors[y2 % len(colors)]
            ax2.set_ylabel(y2_label or f"{var} (a.u.) [ch {y2}]", color=right_color)
            if y2lim:
                ax2.set_ylim(y2lim)
            if xlim:
                ax2.set_xlim(xlim)
            ax2.tick_params(axis="y", colors=right_color)
            ax2.spines["right"].set_color(right_color)

        if show_legend:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
            handles, labels = h1 + h2, l1 + l2

            if legend_loc == "outside":
                ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.15, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                )
            else:
                ax.legend(handles, labels, loc=legend_loc, frameon=True)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        r, c = divmod(j, col_wrap)
        fig.delaxes(axes[r][c])

    if legend_loc == "outside":
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    plt.show()
