from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# package requirements
with open("requirements.txt", encoding='utf-8') as f:
    requirements = [l.strip() for l in f.readlines() if l]

setup(
    name = "croparray",
    version = "0.0.1",
    author = "Tim Stasevich",
    author_email = "Tim.Stasevich@colostate.edu",
    description = ("Python module for for creating and manipulating an array of crops (or regions of interest) from images obtained using single-molecule microscopy."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "BSD 3-Clause License",
    keywords = "single-molecule image processing",
    url = "https://github.com/Colorado-State-University-Stasevich-Lab/croparray",
    package_dir ={'':'croparray'},
    packages=find_packages(where="src"),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8"
    ],
    python_requires='>=3.7'
)
