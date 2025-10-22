import platform

from setuptools import setup


def is_windows():
    return platform.system() == "Windows"


def get_binaries():
    if not is_windows():
        return None
    libs = [
        "onnxruntime.dll",
        "sherpa-onnx-c-api.dll",
        "sherpa-onnx-cxx-api.dll",
        "sherpa-onnx-c-api.lib",
        "sherpa-onnx-cxx-api.lib",
    ]
    prefix = "./sherpa_onnx/lib"
    return [f"{prefix}/{lib}" for lib in libs]


setup(
    name="sherpa-onnx-core",
    version="1.12.15",
    description="Core shared libraries for sherpa-onnx",
    packages=["sherpa_onnx"],
    include_package_data=True,
    data_files=[("Scripts", get_binaries())] if get_binaries() else None,
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
