from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# package requirements
with open("requirements.txt", encoding='utf-8') as f:
    requirements = [l.strip() for l in f.readlines() if l]

# package version
__version__ = None
with open('croparray/__init__.py', encoding='utf-8') as f:
    for row in f:
        if row.startswith('__version__'):
            __version__ = row.strip().split()[-1][1:-1]
            break
        
setup(
    name = "croparray",
    version = __version__,
    author = "Tim Stasevich",
    author_email = "Tim.Stasevich@colostate.edu",
    description = ("Python module for for creating and manipulating an array of crops (or regions of interest) from images obtained using single-molecule microscopy."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "BSD 3-Clause License",
    keywords = "Single-molecule image processing",
    url = "https://github.com/Colorado-State-University-Stasevich-Lab/croparray",
    package_dir = {'croparray':'croparray'},
    packages=find_packages(exclude=['docs','database','notebooks','__pycache__','.gitignore']),
    include_package_data=True,
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
