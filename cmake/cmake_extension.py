# cmake/cmake_extension.py
# Copyright (c)  2023  Xiaomi Corporation
#
# flake8: noqa

import os
import platform
import shutil
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


def is_for_pypi():
    ans = os.environ.get("SHERPA_ONNX_IS_FOR_PYPI", None)
    return ans is not None


def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # In this case, the generated wheel has a name in the form
            # sherpa-xxx-pyxx-none-any.whl
            if is_for_pypi() and not is_macos():
                self.root_is_pure = True
            else:
                # The generated wheel has a name ending with
                # -linux_x86_64.whl
                self.root_is_pure = False

except ImportError:
    bdist_wheel = None


def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):
        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        out_bin_dir = Path(self.build_lib).parent / "sherpa_onnx" / "bin"
        install_dir = Path(self.build_lib).resolve() / "sherpa_onnx"

        sherpa_onnx_dir = Path(__file__).parent.parent.resolve()

        cmake_args = os.environ.get("SHERPA_ONNX_CMAKE_ARGS", "")
        make_args = os.environ.get("SHERPA_ONNX_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")

        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        extra_cmake_args = f" -DCMAKE_INSTALL_PREFIX={install_dir} "
        extra_cmake_args += " -DBUILD_SHARED_LIBS=ON "

        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_CHECK=OFF "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_PYTHON=ON "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_PORTAUDIO=ON "
        extra_cmake_args += " -DSHERPA_ONNX_ENABLE_WEBSOCKET=ON "

        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"

        cmake_args += extra_cmake_args

        if is_windows():
            build_cmd = f"""
         cmake {cmake_args} -B {self.build_temp} -S {sherpa_onnx_dir}
         cmake --build {self.build_temp} --target install --config Release -- -m
            """
            print(f"build command is:\n{build_cmd}")
            ret = os.system(
                f"cmake {cmake_args} -B {self.build_temp} -S {sherpa_onnx_dir}"
            )
            if ret != 0:
                raise Exception("Failed to configure sherpa")

            ret = os.system(
                f"cmake --build {self.build_temp} --target install --config Release -- -m"  # noqa
            )
            if ret != 0:
                raise Exception("Failed to build and install sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export SHERPA_ONNX_MAKE_ARGS="-j"; python setup.py install')
                print('Setting make_args to "-j4"')
                make_args = "-j4"

            build_cmd = f"""
                cd {self.build_temp}

                cmake {cmake_args} {sherpa_onnx_dir}

                make {make_args} install/strip
            """
            print(f"build command is:\n{build_cmd}")

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild sherpa-onnx failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/k2-fsa/sherpa-onnx/issues/new\n"  # noqa
                )

        suffix = ".exe" if is_windows() else ""
        # Remember to also change setup.py

        binaries = ["sherpa-onnx"]
        binaries += ["sherpa-onnx-offline"]
        binaries += ["sherpa-onnx-microphone"]
        binaries += ["sherpa-onnx-microphone-offline"]
        binaries += ["sherpa-onnx-online-websocket-server"]
        binaries += ["sherpa-onnx-offline-websocket-server"]
        binaries += ["sherpa-onnx-online-websocket-client"]
        binaries += ["sherpa-onnx-vad-microphone"]
        binaries += ["sherpa-onnx-offline-tts"]

        if is_windows():
            binaries += ["kaldi-native-fbank-core.dll"]
            binaries += ["sherpa-onnx-c-api.dll"]
            binaries += ["sherpa-onnx-core.dll"]
            binaries += ["sherpa-onnx-portaudio.dll"]
            binaries += ["onnxruntime.dll"]
            binaries += ["kaldi-decoder-core.dll"]
            binaries += ["sherpa-onnx-fst.dll"]
            binaries += ["sherpa-onnx-kaldifst-core.dll"]

        for f in binaries:
            suffix = "" if "dll" in f else suffix
            src_file = install_dir / "bin" / (f + suffix)
            if not src_file.is_file():
                src_file = install_dir / "lib" / (f + suffix)
            if not src_file.is_file():
                src_file = install_dir / ".." / (f + suffix)
            print(f"Copying {src_file} to {out_bin_dir}/")
            shutil.copy(f"{src_file}", f"{out_bin_dir}/")

        shutil.rmtree(f"{install_dir}/bin")
        if is_windows():
            shutil.rmtree(f"{install_dir}/lib")
