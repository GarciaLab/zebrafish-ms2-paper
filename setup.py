from setuptools import find_packages, setup
from pathlib import Path


# Package meta-data.
NAME = "zebrafish_ms2_paper"
DESCRIPTION = "Code for the simulations and plots in the paper by Eck, Moretti, and Schlomann, bioRxiv (2024)."
URL = "https://github.com/bschloma/zebrafish-ms2-paper"
EMAIL = "bschloma@berkeley.edu"
AUTHOR = "Brandon Schlomann"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.1"

long_description = (Path(__file__).parent / "README.md").read_text()

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy>=1.20",
    "scipy>=1.8.0",
    "pandas",
    "matplotlib",
]

# What packages are optional?
EXTRAS = {}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="BSD 3-Clause",
    license_file="LICENSE.txt",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
    ],
)
