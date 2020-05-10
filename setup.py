from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README = (HERE/"README.md").read_text()

setup(
  name="fitloop",
  version="0.1.2",
  description="fitloop trains Pytorch models",
  packages=find_packages(),
  long_description=README,
  long_description_content_type="text/markdown",
  license="MIT",
  author="Alan",
  author_email="2.alan.tom@gmail.com",
  url="https://github.com/18alantom/fitloop",
  classifiers= [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: OS Independent",
    "Framework :: Jupyter"
  ],
  include_package_data=True,
  install_requires=[
    "torch~=1.4",
    "matplotlib~=3.2",
    "numpy~=1.18",
    "tqdm~=4.40"
  ]
)
