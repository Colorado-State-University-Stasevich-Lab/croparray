"""
crop_ops.measure

Functions in crop_ops operate on a *single crop*:

- Input:  2D numpy array (y, x)
- Output: numpy array or scalar

Rules:
- Pure functions: no dataset mutation
- No reliance on global state
- Heavy dependencies (e.g., trackpy) should be imported inside functions
"""

from __future__ import annotations

def spot_detect_and_qc(img, minmass=6000, size=5):
    """
    Locates features in an image using trackpy locate function and creates a new image with a pixel at the location of the feature closest to the center. This pixel has the signal value of the feature.

    This function is used to identify and highlight the location of features in an image. This is useful for visualizing the features and their signal values from an croparray dataset with padding. This function can be used to identify or quality control the features in the image such as mRNA and translation spots.

    Parameters:
    img (numpy.ndarray): The input image.
    minmass (int, optional): The minimum integrated brightness. Defaults to 6000.
    size (int, optional): The size of the features in pixels. Defaults to 5.

    Returns:
    numpy.ndarray: A new image of the same size as the input image, with a pixel at the location of the feature closest to the center. The pixel value is the signal value of the feature.

    """
    import trackpy as tp
    import numpy as np
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No maxima survived mass- and size-based filtering.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="All local maxima were in the margins.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message="Image is completely black.*",
            category=UserWarning,
            module=r"trackpy\..*",
        )
        features = tp.locate(img, size, minmass)

    new_img = np.zeros_like(img)
    if len(features) > 0:
        # Calculate the center of the image
        center_x = img.shape[1] / 2
        center_y = img.shape[0] / 2

        if len(features) > 1:
            # Calculate the Euclidean distance from each feature to the center
            distances = np.sqrt((features['x'] - center_x)**2 + (features['y'] - center_y)**2)
            # Find the index of the feature with the smallest distance
            closest_index = np.argmin(distances)
            x_value = features['x'].values[closest_index]
            y_value = features['y'].values[closest_index]
            signal_value = features['signal'].values[closest_index]
        else:
            x_value = features['x'].values[0]
            y_value = features['y'].values[0]
            signal_value = features['signal'].values[0]

        # Set the pixel at (x_value, y_value) to the maximum pixel value
        new_img[int(y_value), int(x_value)] = signal_value
    return new_img

def binarize_crop_manual(
    img,
    *,
    th1: float,
    th2: float,
    seed: str = "spot",          # "spot" or "center"
    minmass: float = 100.0,      # used only if seed="spot"
    size: int = 5,               # used only if seed="spot"
    max_dist_px: float | None = None,  # optional: reject if nearest component too far
    fill_holes: bool = False,
    close_px: int = 0,           # optional morphological closing radius
    return_uint8: bool = True,
):
    """
    Manual band-pass binarization: keep pixels with th1 <= I <= th2, then keep
    only the connected component nearest the seed (spot or crop center).

    Parameters
    ----------
    img : 2D array
        Crop image.
    th1, th2 : float
        Lower/upper intensity thresholds (inclusive).
    seed : {"spot","center"}
        Seed selection mode:
          - "spot": use spot_detect_and_qc to find seed (y,x)
          - "center": use crop center as seed
    minmass, size : float, int
        Passed to spot_detect_and_qc when seed="spot".
    max_dist_px : float or None
        If set, return empty mask when nearest component is farther than this.
    fill_holes : bool
        Fill holes in the selected component.
    close_px : int
        If >0, apply binary closing to the band-pass mask before selecting the component.
    return_uint8 : bool
        Return uint8 (0/1) if True, else bool.

    Returns
    -------
    mask : 2D array
        Binary mask.
    """
    import numpy as np

    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"binarize_crop_manual expects 2D image, got {img.shape}")

    lo = float(min(th1, th2))
    hi = float(max(th1, th2))

    # Band-pass
    m = (img >= lo) & (img <= hi)

    # Optional cleanup before labeling (helps tails stay connected)
    if close_px and int(close_px) > 0:
        from scipy.ndimage import binary_closing
        m = binary_closing(m, iterations=int(close_px))

    # Determine seed
    H, W = img.shape
    if seed == "center":
        sy, sx = H // 2, W // 2
    elif seed == "spot":
        # Use spot_detect_and_qc to find a seed; fall back to center if it fails
        try:
            spot_out = spot_detect_and_qc(img, minmass=minmass, size=size)
            arr = np.asarray(spot_out)
            if arr.shape == img.shape:
                flat = int(np.nanargmax(arr))
                sy, sx = np.unravel_index(flat, arr.shape)
            elif arr.size == 2:
                sy, sx = int(np.round(arr[0])), int(np.round(arr[1]))
            else:
                sy, sx = H // 2, W // 2
        except Exception:
            sy, sx = H // 2, W // 2
    else:
        raise ValueError("seed must be 'spot' or 'center'")

    # Connected components
    from skimage.measure import label, regionprops

    lab = label(m, connectivity=2)
    if lab.max() == 0:
        out = np.zeros_like(m, dtype=bool)
        return out.astype(np.uint8) if return_uint8 else out

    # Choose the component whose pixels are closest to the seed
    best_label = None
    best_d2 = None

    for rp in regionprops(lab):
        coords = rp.coords  # (N,2) as (row,col)
        dy = coords[:, 0] - sy
        dx = coords[:, 1] - sx
        d2 = float(np.min(dy * dy + dx * dx))
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_label = rp.label

    # Optional distance rejection
    if max_dist_px is not None and best_d2 is not None:
        if np.sqrt(best_d2) > float(max_dist_px):
            out = np.zeros_like(m, dtype=bool)
            return out.astype(np.uint8) if return_uint8 else out

    out = (lab == best_label)

    if fill_holes:
        from scipy.ndimage import binary_fill_holes
        out = binary_fill_holes(out)

    return out.astype(np.uint8) if return_uint8 else out



def binarize_crop(
    img,
    *,
    # --- spot detection (seed) ---
    minmass: float = 100.0,
    size: int = 5,

    # --- preprocessing on the image used for segmentation ---
    smooth_sigma: float = 1.0,
    subtract_bg: str | None = "gaussian",
    bg_sigma: float = 12.0,

    # --- robust background/statistics ---
    bg_mode: str = "border",  # "border" or "all"
    clip_percentiles: tuple[float, float] | None = (1.0, 99.5),

    # --- threshold selection driven by SNR ---
    # thr = bg + gamma*(spot - bg), where gamma depends on SNR
    snr_low: float = 2.0,       # at/under this SNR: be strict
    snr_high: float = 8.0,      # at/over this SNR: allow more growth (tails)
    gamma_strict: float = 0.75, # threshold closer to spot (smaller region)
    gamma_loose: float = 0.30,  # threshold closer to bg (bigger region; tails)
    min_snr_to_segment: float = 1.5,  # below this: return empty or fallback

    # additional floor/ceiling using sigma
    k_bg: float = 2.0,          # enforce thr >= bg + k_bg*sigma
    k_cap: float = 0.0,         # optional: enforce thr <= spot - k_cap*sigma (0 disables)

    # --- watershed / morphology ---
    use_watershed: bool = True,
    compactness: float = 0.0,
    dilate_fg: int = 0,         # expand fg mask by N iterations (optional)
    fill_holes: bool = False,   # optional cleanup
    min_area_px: int = 0,       # optional: keep only if region area >= this

    # --- fallback behavior ---
    fallback: str = "empty",    # "empty" or "otsu"
    fallback_thr_scale: float = 3.0,

    return_uint8: bool = True,
):
    """
    Segment (binarize) a single 2D crop using a spot-derived seed and
    an SNR-adaptive global threshold, optionally refined by watershed.

    Algorithm
    ---------
    1) Preprocess image (smooth + optional background subtraction)
    2) Detect spot via spot_detect_and_qc -> seed (y,x)
    3) Estimate background (bg) and robust sigma (σ) within crop
    4) Compute SNR = (spot - bg) / σ
    5) Choose gamma based on SNR, then threshold:
           thr = bg + gamma*(spot - bg)
       with a floor thr >= bg + k_bg*σ
    6) Create foreground mask fg = x > thr
    7) Watershed seeded at (y,x) constrained to fg to get a connected region

    Notes
    -----
    - Designed for crops containing a diffraction-limited spot near the center,
      but will include elongated “comet tails” when the spot is bright enough.
    - SNR-adaptive gamma is the main control knob for tail inclusion.
    """
    import numpy as np

    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"binarize_crop expects a 2D image, got shape {img.shape}")

    x = img.astype(np.float32, copy=False)

    # ----------------------------
    # Preprocessing
    # ----------------------------
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
        except Exception as e:
            raise ImportError("scipy is required for smooth_sigma>0") from e
        x = gaussian_filter(x, float(smooth_sigma))

    if subtract_bg is not None:
        mode = str(subtract_bg).lower()
        if mode == "gaussian":
            from scipy.ndimage import gaussian_filter
            x = x - gaussian_filter(x, float(bg_sigma))
        else:
            raise ValueError(f"Unknown subtract_bg mode: {subtract_bg!r}")

    # Optional clipping to stabilize stats/spot response
    if clip_percentiles is not None:
        lo, hi = clip_percentiles
        if not (0 <= lo < hi <= 100):
            raise ValueError("clip_percentiles must be (low, high) with 0 <= low < high <= 100")
        finite = x[np.isfinite(x)]
        if finite.size > 0:
            vmin = np.percentile(finite, lo)
            vmax = np.percentile(finite, hi)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                x = np.clip(x, vmin, vmax)

    H, W = x.shape

    # ----------------------------
    # Spot detection -> seed (y,x)
    # ----------------------------
    # Expectation: spot_detect_and_qc operates on a single crop.
    # We support a few return types robustly:
    #  - ndarray same shape: use argmax
    #  - dict with y/x keys
    #  - tuple/list (y,x)
    try:
        spot_out = spot_detect_and_qc(x, minmass=minmass, size=size)
    except Exception:
        spot_out = None

    seed_y = seed_x = None

    if spot_out is None:
        seed_y = seed_x = None
    elif isinstance(spot_out, dict):
        # allow flexible key names
        for ky in ("y", "spot_y", "cy", "row"):
            if ky in spot_out:
                seed_y = spot_out[ky]
                break
        for kx in ("x", "spot_x", "cx", "col"):
            if kx in spot_out:
                seed_x = spot_out[kx]
                break
    else:
        arr = np.asarray(spot_out)
        if arr.shape == x.shape:
            # Use the strongest response as seed
            try:
                flat_idx = int(np.nanargmax(arr))
                seed_y, seed_x = np.unravel_index(flat_idx, arr.shape)
            except Exception:
                seed_y = seed_x = None
        elif arr.size == 2:
            seed_y, seed_x = arr[0], arr[1]
        else:
            seed_y = seed_x = None

    # If no seed from spot detection, optionally use center as last resort
    if seed_y is None or seed_x is None:
        # fallback behavior
        if fallback == "otsu":
            try:
                from skimage.filters import threshold_otsu
                finite = x[np.isfinite(x)]
                if finite.size == 0:
                    mask = np.zeros_like(x, dtype=bool)
                    return mask.astype(np.uint8) if return_uint8 else mask
                thr = float(threshold_otsu(finite)) * float(fallback_thr_scale)
                mask = x > thr
                return mask.astype(np.uint8) if return_uint8 else mask
            except Exception:
                mask = np.zeros_like(x, dtype=bool)
                return mask.astype(np.uint8) if return_uint8 else mask

        mask = np.zeros_like(x, dtype=bool)
        return mask.astype(np.uint8) if return_uint8 else mask

    # sanitize seed
    try:
        sy = int(np.round(float(seed_y)))
        sx = int(np.round(float(seed_x)))
    except Exception:
        sy = sx = -1

    if not (0 <= sy < H and 0 <= sx < W):
        mask = np.zeros_like(x, dtype=bool)
        return mask.astype(np.uint8) if return_uint8 else mask

    # Spot intensity is taken from the (preprocessed) image at the seed
    spot_val = float(x[sy, sx])

    # ----------------------------
    # Robust background + sigma
    # ----------------------------
    if bg_mode == "border":
        border = np.concatenate([x[0, :], x[-1, :], x[:, 0], x[:, -1]])
        vals = border[np.isfinite(border)]
    elif bg_mode == "all":
        vals = x[np.isfinite(x)].ravel()
    else:
        raise ValueError(f"Unknown bg_mode: {bg_mode!r}")

    if vals.size == 0:
        mask = np.zeros_like(x, dtype=bool)
        return mask.astype(np.uint8) if return_uint8 else mask

    # robust bg = median, robust sigma from MAD
    bg = float(np.median(vals))
    mad = float(np.median(np.abs(vals - bg)))
    sigma = 1.4826 * mad  # robust sigma estimate
    if not np.isfinite(sigma) or sigma <= 1e-6:
        # fallback to std if MAD collapses
        sigma = float(np.std(vals))
        if not np.isfinite(sigma) or sigma <= 1e-6:
            sigma = 1.0

    # SNR
    snr = (spot_val - bg) / sigma

    # If spot is not meaningfully above background, don’t segment aggressively
    if not np.isfinite(snr) or snr < float(min_snr_to_segment):
        if fallback == "otsu":
            try:
                from skimage.filters import threshold_otsu
                finite = x[np.isfinite(x)]
                thr = float(threshold_otsu(finite)) * float(fallback_thr_scale)
                mask = x > thr
                return mask.astype(np.uint8) if return_uint8 else mask
            except Exception:
                pass
        mask = np.zeros_like(x, dtype=bool)
        return mask.astype(np.uint8) if return_uint8 else mask

    # ----------------------------
    # Choose gamma based on SNR
    # Higher SNR -> looser threshold -> capture tails
    # ----------------------------
    if float(snr_high) <= float(snr_low):
        w = 1.0
    else:
        w = (snr - float(snr_low)) / (float(snr_high) - float(snr_low))
        w = float(np.clip(w, 0.0, 1.0))

    # interpolate: at low SNR use strict gamma; at high SNR use loose gamma
    gamma = (1.0 - w) * float(gamma_strict) + w * float(gamma_loose)

    thr = bg + gamma * (spot_val - bg)

    # sigma-based floor (prevents huge masks when bg/spot are close)
    thr = max(thr, bg + float(k_bg) * sigma)

    # optional cap (prevents threshold being too close to spot)
    if k_cap and float(k_cap) > 0:
        thr = min(thr, spot_val - float(k_cap) * sigma)

    # ----------------------------
    # Foreground mask and watershed
    # ----------------------------
    fg = x > thr
    fg[sy, sx] = True  # ensure seed is included

    if dilate_fg and int(dilate_fg) > 0:
        from scipy.ndimage import binary_dilation
        fg = binary_dilation(fg, iterations=int(dilate_fg))

    if not use_watershed:
        mask = fg
    else:
        from skimage.filters import sobel
        from skimage.segmentation import watershed

        elev = sobel(x)  # edges as barriers
        markers = np.zeros((H, W), dtype=np.int32)
        markers[sy, sx] = 1
        labels = watershed(elev, markers=markers, mask=fg, compactness=float(compactness))
        mask = labels == 1

    if fill_holes:
        try:
            from scipy.ndimage import binary_fill_holes
            mask = binary_fill_holes(mask)
        except Exception:
            pass

    if min_area_px and int(min_area_px) > 0:
        if int(mask.sum()) < int(min_area_px):
            mask = np.zeros_like(mask, dtype=bool)

    return mask.astype(np.uint8) if return_uint8 else mask


# def binarize_crop(
#     img,
#     method: str = "otsu",
#     *,
#     # --- new knobs ---
#     threshold: float | None = None,
#     thr_scale: float = 1.0,
#     thr_offset: float = 0.0,
#     # --- preprocessing ---
#     smooth_sigma: float = 1.0,
#     subtract_bg: str | None = None,
#     bg_sigma: float = 12.0,
#     clip_percentiles: tuple[float, float] | None = (1.0, 99.5),
#     # --- postprocessing/behavior ---
#     invert: bool = False,
#     min_threshold: float | None = None,
#     return_uint8: bool = True,
# ):
#     """
#     Binarize a single 2D crop via a global threshold (manual or auto).

#     Parameters
#     ----------
#     img : 2D array (y, x)
#         Single crop image.
#     method : str
#         Auto-threshold method if `threshold` is None. One of:
#         "otsu", "yen", "li", "triangle", "mean", "median".
#     threshold : float or None
#         If provided, use this manual threshold (after preprocessing).
#     thr_scale, thr_offset : float
#         After auto/manual threshold is determined, apply:
#             thr = thr * thr_scale + thr_offset
#         Use thr_scale > 1 to make masks smaller (stricter).
#     smooth_sigma : float
#         Gaussian smoothing sigma (pixels) before thresholding.
#     subtract_bg : {"gaussian", None}
#         If "gaussian", subtract a heavily smoothed background estimate.
#     bg_sigma : float
#         Sigma for background estimate if subtract_bg="gaussian".
#     clip_percentiles : (low, high) or None
#         Percentile clip of the preprocessed image before computing threshold.
#         This stabilizes histogram-based thresholds when there are outliers.
#     invert : bool
#         If True, invert mask.
#     min_threshold : float or None
#         Enforce thr >= min_threshold.
#     return_uint8 : bool
#         If True, return uint8 {0,1}. If False, return bool mask.

#     Returns
#     -------
#     mask : 2D array
#         Binary mask (uint8 0/1 or bool).
#     """
#     import numpy as np

#     img = np.asarray(img)
#     if img.ndim != 2:
#         raise ValueError(f"binarize_crop expects a 2D image, got shape {img.shape}")

#     x = img.astype(np.float32, copy=False)

#     # Optional smoothing
#     if smooth_sigma and smooth_sigma > 0:
#         try:
#             from scipy.ndimage import gaussian_filter
#         except Exception as e:
#             raise ImportError("scipy is required for smooth_sigma>0") from e
#         x = gaussian_filter(x, smooth_sigma)

#     # Optional background subtraction
#     if subtract_bg is not None:
#         mode = str(subtract_bg).lower()
#         if mode == "gaussian":
#             try:
#                 from scipy.ndimage import gaussian_filter
#             except Exception as e:
#                 raise ImportError("scipy is required for subtract_bg='gaussian'") from e
#             bg = gaussian_filter(x, bg_sigma)
#             x = x - bg
#         else:
#             raise ValueError(f"Unknown subtract_bg mode: {subtract_bg!r}")

#     # Optional percentile clipping (robustify histogram)
#     if clip_percentiles is not None:
#         lo, hi = clip_percentiles
#         if not (0 <= lo < hi <= 100):
#             raise ValueError("clip_percentiles must be (low, high) with 0 <= low < high <= 100")
#         finite = x[np.isfinite(x)]
#         if finite.size > 0:
#             vmin = np.percentile(finite, lo)
#             vmax = np.percentile(finite, hi)
#             if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
#                 x = np.clip(x, vmin, vmax)

#     # Determine threshold (manual or auto)
#     if threshold is not None:
#         thr = float(threshold)
#     else:
#         m = method.lower().strip()
#         finite = x[np.isfinite(x)]
#         if finite.size == 0:
#             thr = 0.0
#         elif m in ("otsu", "yen", "li", "triangle"):
#             try:
#                 from skimage.filters import (
#                     threshold_otsu,
#                     threshold_yen,
#                     threshold_li,
#                     threshold_triangle,
#                 )
#             except Exception as e:
#                 raise ImportError("scikit-image is required for method in {'otsu','yen','li','triangle'}") from e

#             if m == "otsu":
#                 thr = float(threshold_otsu(finite))
#             elif m == "yen":
#                 thr = float(threshold_yen(finite))
#             elif m == "li":
#                 thr = float(threshold_li(finite))
#             else:  # triangle
#                 thr = float(threshold_triangle(finite))

#         elif m == "mean":
#             thr = float(np.mean(finite))
#         elif m == "median":
#             thr = float(np.median(finite))
#         else:
#             raise ValueError(f"Unknown method {method!r}")

#     # Apply user knobs
#     thr = thr * float(thr_scale) + float(thr_offset)

#     if min_threshold is not None:
#         thr = max(thr, float(min_threshold))

#     mask = x > thr
#     if invert:
#         mask = ~mask

#     if return_uint8:
#         return mask.astype(np.uint8)
#     return mask

