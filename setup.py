#!/usr/bin/env python

from setuptools import setup

setup(
    name="uncert",
    version="0.1.0",
    description="Simple arithmetic and displaying of measurements and uncertainties",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zhang Maiyun",
    author_email="maz005@ucsd.edu",
    url="https://github.com/myzhang1029/uncert",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    packages=["uncert"],
    install_requires=["numpy"],
)
