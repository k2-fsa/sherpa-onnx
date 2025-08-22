import sys
from . import _info


def main():
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: python3 -m sherpa_onnx [--cflags|--c-api-libs|--c-api-libs-only-L|--c-api-libs-only-l|--cxx-api-libs|--cxx-api-libs-only-L|--cxx-api-libs-only-l]"
        )
        sys.exit(1)

    if "--cflags" in args:
        print(f"-I{_info.get_include_dir()}")
    elif "--c-api-libs" in args:
        lib_flags = " ".join(f"-l{lib}" for lib in _info.get_c_api_libs())
        print(f"-L{_info.get_libs_dir()} {lib_flags}")
    elif "--c-api-libs-only-L" in args:
        print(f"-L{_info.get_libs_dir()}")
    elif "--c-api-libs-only-l" in args:
        print(" ".join(f"-l{lib}" for lib in _info.get_c_api_libs()))
    elif "--cxx-api-libs" in args:
        lib_flags = " ".join(f"-l{lib}" for lib in _info.get_cxx_api_libs())
        print(f"-L{_info.get_libs_dir()} {lib_flags}")
    elif "--cxx-api-libs-only-L" in args:
        print(f"-L{_info.get_libs_dir()}")
    elif "--cxx-api-libs-only-l" in args:
        print(" ".join(f"-l{lib}" for lib in _info.get_cxx_api_libs()))
    else:
        print("Unknown option:", args[0])
        sys.exit(1)


if __name__ == "__main__":
    main()
