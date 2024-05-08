from setuptools import setup, find_packages

setup(
    name="thispackage",
    version="0.1.0",
    packages=find_packages(where="src"),  # Assuming your code is in the src directory
    package_dir={"": "src"},  # Tell setuptools that package modules are under src
)
