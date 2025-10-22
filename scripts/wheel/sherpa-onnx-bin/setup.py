import glob
import platform

from setuptools import setup


def is_windows():
    return platform.system() == "Windows"


bin_files = glob.glob("bin/*")
print("bin_files", bin_files)

setup(
    name="sherpa-onnx-bin",
    version="1.12.15",
    description="Binary executables for sherpa-onnx",
    author="The sherpa-onnx development team",
    url="https://github.com/k2-fsa/sherpa-onnx",
    author_email="dpovey@gmail.com",
    zip_safe=False,
    license="Apache 2.0",
    packages=[],
    data_files=[("Scripts", bin_files) if is_windows() else ("bin", bin_files)],
    install_requires=[
        "sherpa-onnx-core==1.12.15",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
