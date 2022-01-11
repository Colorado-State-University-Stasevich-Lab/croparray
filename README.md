# croparray
Authors: Tim Stasevich.

## Description
This module is intended for creating and manipulating an array of crops (or regions of interest) that were generated from a multicolor TIF video obtained from single-molecule microscopy.

<img src= https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Fig1-CropArrayConceptV4.png alt="drawing" width="600"/>


## Documentation 
* Documentation is accessible via [croparray.readthedocs](https://croparray.readthedocs.io) 

## Colab implementation

 * Implementation in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1Ru-_ak9PpW9bGM_H9SlffmdDQOxG16-4?usp=sharing)

<img src= https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Croparray.gif alt="drawing" width="1000"/>

## Local installation from the Github repository

* Install [anaconda](https://anaconda.org).

* Clone the Github repository
```bash
    git clone https://github.com/Colorado-State-University-Stasevich-Lab/croparray.git
```

* To create a virtual environment navigate to your local repository and use:
```bash
    conda create -n croparray_env python=3.8 -y
    source activate croparray_env
```

* To install the rest of requirements use:
```
    pip install -r requirements.txt
```

## Local installation using PIP

* To create a virtual environment using:

```bash
    conda create -n croparray_env python=3.8 -y
    source activate croparray_env
```

* Open the terminal and use [pip](https://pip.pypa.io/en/stable/) for the installation:
```bash
    pip install croparray
```

## Deactivating and removing the environment

* To deactivate or remove the environment from your computer use:
```bash
    conda deactivate
```
* To remove the environment use:
```bash
    conda env remove -n croparray_env
```
* To unistall croparray use
```bash
    pip uninstall croparray
```

## additional troubleshooting information
* If you cannot see the package installed on your computer, try using ```pip3```. For example: 
```bash
    pip3 install croparray
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
- License for [Napari](https://github.com/napari/napari): BSD-3-Clause License. Copyright (c) 2018, Napari. All rights reserved.

- License for [xarray](https://github.com/pydata/xarray): Apache License. Version 2.0, January 2004. Copyright 2014-2019, xarray Developers
