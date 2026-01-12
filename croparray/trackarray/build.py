import numpy as np
import pandas as pd
import xarray as xr

# Make track_id dimension in crop array (that has been tracked)
def track_array(ca):
    """
    Create a track-array dataset from a tracked crop-array dataset by grouping
    entries by unique track IDs (stored in ca['id']).

    This version FIXES the common issue where `fov` becomes a non-index
    coordinate on `t` (i.e., `fov (t)`), which breaks `.sel(fov=...)` on
    variables like `best_z`. We enforce `fov` as a true dimension (length 1)
    per track, under the assumption that each track belongs to exactly one FOV.

    Parameters
    ----------
    ca : xarray.Dataset
        Crop-array dataset that contains an `id` variable with per-(fov,n,t)
        track IDs, and has a stacked dimension named 'stacked_fov_n_t'
        (created by your crop-array pipeline).

    Returns
    -------
    xarray.Dataset
        Track-array dataset with dimension `track_id` and (typically) `fov`
        as a real dimension. Variables are aligned across tracks with fill_value=0.
    """
    # Find all unique (non-zero) track IDs
    my_ids = np.unique(ca["id"].values)
    my_ids = my_ids[my_ids != 0]

    my_das = []
    for tid in my_ids:
        # Select the group for this track id and convert stacked index -> time index
        temp = (
            ca.groupby("id")[tid]
              .reset_index("stacked_fov_n_t")
              .sortby("t")
              .reset_coords("n", drop=True)                 # n not meaningful in track view
              .set_index(stacked_fov_n_t="t")               # make 't' the index for this track
              .rename({"stacked_fov_n_t": "t"})
        )

        # ---- FIX: ensure `fov` is a true dimension, not a coord on `t` ----
        if "fov" in temp.coords:
            fovs = np.unique(temp["fov"].values)
            if len(fovs) != 1:
                raise ValueError(f"Track {tid} spans multiple FOVs: {fovs}")
            fov0 = int(fovs[0])

            # Remove the (t)-coordinate and promote to a real dimension
            temp = temp.drop_vars("fov")
            temp = temp.expand_dims(fov=[fov0])

        my_das.append(temp)

    # Concatenate into track array
    my_taz = xr.concat(my_das, dim=pd.Index(my_ids, name="track_id"), fill_value=0)

    # Reorder dimensions (drop 'n' from transpose; it was dropped above)
    my_taz = my_taz.transpose("track_id", "fov", "t", "z", "y", "x", "ch", missing_dims="ignore")

    return my_taz