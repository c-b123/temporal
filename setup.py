from setuptools import setup, find_packages

from temporal.my_pip_package import __version__

setup(
    name='temporal',
    version=__version__,
    author='chris',

    packages=find_packages()
)
