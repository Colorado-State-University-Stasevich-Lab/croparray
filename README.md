# croparray
Authors: Tim Stasevich.

## Description
This module is intended for creating and manipulating an array of crops (or regions of interest) that were generated from a multicolor TIF video obtained from single-molecule microscopy.

<img src= https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Fig1-CropArrayConceptV2.png alt="drawing" width="600"/>

## Installation

Open the terminal and use [pip](https://pip.pypa.io/en/stable/) for the installation:
```bash
pip install croparray
```

## Usage
* Visualizes single-molecule objects locaded on super-resolution microscope images.
* Calculates the best maximim projection from a 3D array.
* Measures intensity on the detected objects.
* Calculates the correlation between two equal-length, 1D signals.
* Creates and saves a crop array video at output_direction/output_filename from a 3D tif video (video_3D) and corresponding track dataframe (tracks).
* Creates and saves a particle array video at output_direction/output_filename from a 3D tif video (video_3D) and corresponding particle array dataframe (particles). 


## Licenses for dependencies
- License for [Napari](https://github.com/napari/napari): BSD-3-Clause.

- License for [xarray](https://github.com/pydata/xarray): Copyright 2014-2019, xarray Developers

