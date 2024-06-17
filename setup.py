from setuptools import setup, find_packages

from temporal.my_pip_package import __version__

setup(
    name='temporal',
    version=__version__,
    author='chris',
    url='https://github.com/c-b123/temporal',
    author_email='christian.berger2@uzh.ch',
    python_requires=">=3.11",
    packages=find_packages()
)
