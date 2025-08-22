from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="sherpa-onnx-core",
    version="1.12.9",
    description="Core shared libraries for sherpa-onnx",
    packages=["sherpa_onnx"],
    include_package_data=True,
    author="The sherpa-onnx development team",
    url="https://github.com/k2-fsa/sherpa-onnx",
    author_email="dpovey@gmail.com",
    zip_safe=False,
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
