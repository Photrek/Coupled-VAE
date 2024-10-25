import os
from setuptools import setup, find_packages

PACKAGE_NAME = "coupledvae"

here = os.path.abspath(os.path.dirname(__file__))
info = {}
with open(os.path.join(here, PACKAGE_NAME, '__version__.py'), 'r') as f:
    exec(f.read(), info)
with open("README.md", "r") as fh:
    READ_ME = fh.read()


setup(
    name=info['__title__'],
    author=info['__author__'],
    description=info['__description__'],
    url=info['__url__'],
    version=info['__version__'],
    license=info['__license__'],
    long_description=READ_ME,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    python_requires='>= 3.6',
    install_requires=[
        "pandas", "tensorflow", "numpy", "tqdm", "nsc", "scikit-learn", "matplotlib", "pathlib", "tensorflow_datasets"
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: GNU General Public License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Operating System :: OS Independent",
    ],
)
