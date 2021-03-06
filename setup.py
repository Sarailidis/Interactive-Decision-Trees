# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:30:55 2021

@author: Georgios Sarailidis
"""


# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

from io import open

# =============================================================================
# here = path.abspath(path.dirname(__file__))
# # Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()
# =============================================================================

setup(
    name='InteractiveDT',  # Required
    version='1.0.0', # Required
    description='Toolbox for Interactive Construction and analysis of Decision Trees',  # Optional
    url='https://github.com/Sarailidis/Interactive-Decision-Trees',  # Optional
    author='Georgios Sarailidis',  # Optional
    author_email='g.sarailidis@bristol.ac.uk',  # Optional
    packages=find_packages(exclude=['Workflows']),  # Required.
    install_requires=[
        "scikit-learn==0.24.1",
        "plotly>=4.5.2",
        "ipywidgets>=7.5.1",
        "python-igraph>=0.8.0",
        "chart-studio>=1.0.0",
        "pandas",
        "numpy",
        "matplotlib"
    ],
)




# =============================================================================
# "scikit-learn==0.22.1",
# "plotly==4.5.2",
# "ipywidgets==7.5.1",
# "python-igraph==0.8.0",
# "chart-studio==1.0.0",
# "pandas==0.25.2",
# "numpy==1.16.5",
# "matplotlib==3.1.1"
# =============================================================================



# =============================================================================
# "scikit-learn==0.22.1",
# "plotly==4.7.1",
# "ipywidgets==7.5.1",
# "python-igraph==0.8.2",
# "chart-studio==1.1.0",
# "pandas==1.2.4",
# "numpy==1.18.1",
# "matplotlib==3.2.1"
# =============================================================================
