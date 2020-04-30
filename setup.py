# -------------------------------------------

# Created by:               jasper
# as part of the project:   Bachelorarbeit
# Date:                     4/29/20

#--------------------------------------------


import setuptools
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name="beampropagtor.py-m4rtins",
    version="0.7",
    author="Jasper Martins",
    author_email="m.jasper.martins@gmail.com",
    description="This package provides methods to setup and perform 2D FD-Beampropagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)