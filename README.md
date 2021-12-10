# croparray
Authors: Tim Stasevich.

## Description
This module is intended for creating and manipulating an array of crops (or regions of interest) that were generated from a multicolor TIF video obtained from single-molecule microscopy.
4
<img src= https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Fig1-CropArrayConceptV.png alt="drawing" width="600"/>

## Installation

Open the terminal and use [pip](https://pip.pypa.io/en/stable/) for the installation:
```bash
pip install croparray
```

## Usage
* Organizes crops and measurements of spots of interest from tif images in a convenient x-array format for reduced filesize and more open and reproducible analyses.
* Visualizes crops of detected spots from super-resolution microscope images.
* Calculates the best maximum projection for each crop containing a detected spot.
* Measures intensity of detected spots within crops.
* Calculates the correlation between two equal-length, 1D signals.
* Saves the crop array as a netcdf file at output_direction/output_filename.
* Integrates with Napari for fast and convenient review of crops of detected spots.


## Licenses for dependencies
- License for [Napari](https://github.com/napari/napari): BSD-3-Clause.

- License for [xarray](https://github.com/pydata/xarray): Copyright 2014-2019, xarray Developers

