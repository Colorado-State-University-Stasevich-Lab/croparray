# croparray
Authors: Tim Stasevich.

## Description
This module is intended for creating and manipulating an array of crops (or regions of interest) that were generated from a multicolor TIF video obtained from single-molecule microscopy.

<img src="https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Fig1-CropArrayConceptV4.png" alt="drawing" width="600"/>

## Documentation
* Documentation is accessible via [croparray.readthedocs](https://croparray.readthedocs.io)

## Colab implementation
* Implementation in Google Colab  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ru-_ak9PpW9bGM_H9SlffmdDQOxG16-4?usp=sharing)

<img src="https://github.com/Colorado-State-University-Stasevich-Lab/croparray/raw/main/docs/images/Croparray.gif" alt="drawing" width="1000"/>

---

## Local installation from the GitHub repository (Recommended)

This is the **recommended and most reliable installation method**, especially if you plan to use Napari.

### 1. Install Conda
Install Anaconda or Miniconda from:  
https://www.anaconda.com

### 2. Clone the GitHub repository
```bash
git clone https://github.com/Colorado-State-University-Stasevich-Lab/croparray.git
cd croparray
```

### 3. Create the environment from `environment.yml`
```bash
conda env create -f environment.yml
conda activate croparray_env
```
> **Note on Napari compatibility**  
> The provided `environment.yml` pins Napari to a version range (`>=0.6,<0.7`) that is known to be compatible with Python 3.10 and stable under Linux/WSL.  
> If you modify the environment or Python version, ensure that Napari remains compatible.


### 4. Install croparray (editable mode)
```bash
python -m pip install -e . --no-deps
```

### 5. (Optional) Register the Jupyter kernel (recommended for VS Code / notebooks)
```bash
python -m ipykernel install --user --name croparray_env --display-name "Python (croparray_env)"
```

---
### 6. Verify the installation
```bash
python -c "import croparray, napari; print('croparray OK, napari', napari.__version__)"
```


## Usage

* Organizes crops and measurements of spots of interest from tif images in a convenient xarray format for reduced file size and more open and reproducible analyses.
* Visualizes crops of detected spots from super-resolution microscope images.
* Calculates the best maximum projection for each crop containing a detected spot.
* Measures intensity of detected spots within crops.
* Calculates the correlation between two equal-length, 1D signals.
* Saves the crop array as a NetCDF file at `output_directory/output_filename`.
* Integrates with Napari for fast and convenient review of crops of detected spots.

---

## Deactivating and removing the environment

* Deactivate the environment:
```bash
conda deactivate
```

* Remove the environment:
```bash
conda env remove -n croparray_env
```

* Uninstall croparray:
```bash
pip uninstall croparray
```

---

## Licenses for dependencies

- License for [Napari](https://github.com/napari/napari): BSD-3-Clause License.  
  Copyright (c) 2018, Napari.

- License for [xarray](https://github.com/pydata/xarray): Apache License 2.0.  
  Copyright 2014–2019, xarray Developers.

