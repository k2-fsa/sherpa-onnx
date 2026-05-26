#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation

"""
Download all third-party dependencies for sherpa-onnx.

Reads URLs and SHA256 checksums from cmake/*.cmake files and downloads them
to a local directory for offline builds.

Usage:
    python3 cmake/download-all-deps.py

The script will:
  1. Discover 16 common (platform-independent) dependencies by parsing
     cmake files (asio, eigen, openfst, pybind11, etc.)
  2. Optionally download onnxruntime for a specific platform
     (OS, architecture, library type, build type)
  3. Download all selected files with SHA256 verification
  4. Skip files that already exist with the correct hash
  5. Retry failed downloads up to 3 times, then try mirror URLs

Where to place downloaded files:
  The cmake build system searches for pre-downloaded files in these
  locations (in order):

    ~/Downloads/<filename>          # default download directory
    <project_root>/<filename>       # next to CMakeLists.txt
    <build_dir>/<filename>          # in the build directory
    /tmp/<filename>                 # system temp directory

  You can place all downloaded files in any ONE of these directories.
  The default download directory is ~/Downloads, which works without
  any extra cmake flags.

  To use a custom directory, either:
    - Download to ~/Downloads (the default), or
    - Move the files to the project root or build directory, or
    - Pass -DCMAKE_SOURCE_DIR=<path> or set TMP/TEMP env variable

Example: offline build workflow
  # On a machine with internet:
  python3 cmake/download-all-deps.py
  # Select platform, download to ~/Downloads

  # Transfer ~/Downloads/*.tar.gz, *.zip, etc. to the target machine
  scp ~/Downloads/*.tar* user@target:~/Downloads/

  # On the target machine (no internet needed):
  cd sherpa-onnx && mkdir build && cd build
  cmake -DSHERPA_ONNX_ENABLE_GPU=OFF ..
  make -j4

  CMake will find the pre-downloaded files in ~/Downloads and skip
  fetching them from the internet.
"""

import hashlib
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def get_cmake_dir():
    """Return the cmake/ directory relative to this script."""
    return Path(__file__).resolve().parent


def get_hardcoded_deps():
    """Return deps that don't have a cmake file (e.g., kaldifst).

    Returns list of (name, url, url2, sha256).
    """
    return [
        (
            "kaldifst",
            "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.8.0.tar.gz",
            "",
            "3f247b7e5a2409071202f5e2bc6200060f66728c0a3443c03923ad2723e040b3",
        ),
    ]


def parse_url_and_hash(filepath):
    """Parse a cmake file for URL and SHA256 hash pairs.

    Returns a list of (name, url, url2, sha256) tuples.
    """
    text = filepath.read_text()
    deps = []

    # Pattern: set(name_URL  "https://...")
    #          set(name_URL2 "https://...")  (optional)
    #          set(name_HASH "SHA256=...")
    # Use [\w-]+ to match names with hyphens (e.g., simple-sentencepiece)
    name_re = r'([\w-]+?)'
    url_pattern = re.compile(
        rf'set\({name_re}_URL\s+"([^"]+)"\)', re.MULTILINE
    )
    url2_pattern = re.compile(
        rf'set\({name_re}_URL2\s+"([^"]+)"\)', re.MULTILINE
    )
    hash_pattern = re.compile(
        rf'set\({name_re}_HASH\s+"SHA256=([^"]+)"\)', re.MULTILINE
    )

    # Keep first valid (non-variable) URL for each name
    urls = {}
    for m in url_pattern.finditer(text):
        name, url = m.group(1), m.group(2)
        if "${" not in url and name not in urls:
            urls[name] = url

    url2s = {m.group(1): m.group(2) for m in url2_pattern.finditer(text)}
    hashes = {m.group(1): m.group(2) for m in hash_pattern.finditer(text)}

    for name, url in urls.items():
        sha256 = hashes.get(name, "")
        url2 = url2s.get(name, "")
        if sha256:
            deps.append((name, url, url2, sha256))

    return deps


def parse_windows_onnxruntime(filepath, crt, build_type):
    """Parse a Windows onnxruntime cmake file for a specific CRT + build type.

    Returns (url, sha256) or (None, None).
    """
    text = filepath.read_text()

    # Find the hash variable: ONNXRUNTIME_HASH_{crt}_{build_type}
    hash_var = f"ONNXRUNTIME_HASH_{crt}_{build_type}"
    hash_pattern = re.compile(
        rf'set\({hash_var}\s+"SHA256=([^"]+)"\)', re.MULTILINE
    )
    m = hash_pattern.search(text)
    if not m:
        return None, None
    sha256 = m.group(1)

    # Find the filename template and resolve it
    filename_pattern = re.compile(
        r'set\(onnxruntime_filename\s+"([^"]+)"\)', re.MULTILINE
    )
    m = filename_pattern.search(text)
    if not m:
        return None, None
    filename_tpl = m.group(1)
    filename = filename_tpl.replace("${onnxruntime_crt}", crt)
    filename = filename.replace("${CMAKE_BUILD_TYPE}", build_type)

    # Find the URL - take first match, skip cmake variable references
    url_pattern = re.compile(
        r'set\(onnxruntime_URL\s+"([^"]+)"\)', re.MULTILINE
    )
    for m in url_pattern.finditer(text):
        url = m.group(1)
        if "${" not in url or "${onnxruntime_filename}" in url:
            # Resolve filename variable
            url = url.replace("${onnxruntime_filename}", filename)
            return url, sha256

    return None, None


def discover_common_deps(cmake_dir):
    """Discover platform-independent deps from cmake files.

    Returns list of (name, url, url2, sha256).
    """
    # These cmake files contain platform-independent deps
    dep_files = [
        "asio.cmake",
        "cargs.cmake",
        "eigen.cmake",
        "espeak-ng-for-piper.cmake",
        "googletest.cmake",
        "hclust-cpp.cmake",
        "json.cmake",
        "kaldi-decoder.cmake",
        "kaldi-native-fbank.cmake",
        "openfst.cmake",
        "piper-phonemize.cmake",
        "portaudio.cmake",
        "pybind11.cmake",
        "simple-sentencepiece.cmake",
        "websocketpp.cmake",
    ]

    all_deps = []
    for f in dep_files:
        path = cmake_dir / f
        if path.exists():
            deps = parse_url_and_hash(path)
            all_deps.extend(deps)
    return all_deps


def select_menu(title, options):
    """Display a numbered menu and return the selected option."""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = input(f"Enter choice [1-{len(options)}]: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except (ValueError, EOFError):
            pass
        print("Invalid choice, try again.")


def ask_yes_no(question, default="y"):
    """Ask a yes/no question."""
    suffix = " [Y/n]: " if default == "y" else " [y/N]: "
    while True:
        try:
            answer = (input(question + suffix).strip().lower() or default)
        except EOFError:
            answer = default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def select_onnxruntime_deps(cmake_dir):
    """Interactively select onnxruntime deps based on platform.

    Returns list of (name, url, url2, sha256).
    """
    os_choice = select_menu(
        "Select target OS:",
        ["Linux", "macOS", "Windows", "WASM"],
    )

    if os_choice == "WASM":
        path = cmake_dir / "onnxruntime-wasm-simd.cmake"
        deps = parse_url_and_hash(path)
        return deps

    if os_choice == "Linux":
        arch = select_menu(
            "Select architecture:",
            ["x86_64", "aarch64", "arm", "riscv64"],
        )

        if arch == "riscv64":
            variant = select_menu(
                "Select variant:",
                ["shared", "static", "spacemit"],
            )
            if variant == "spacemit":
                path = cmake_dir / "onnxruntime-linux-riscv64-spacemit.cmake"
            elif variant == "static":
                path = cmake_dir / "onnxruntime-linux-riscv64-static.cmake"
            else:
                path = cmake_dir / "onnxruntime-linux-riscv64.cmake"
            return parse_url_and_hash(path)

        if arch == "arm":
            lib_type = select_menu(
                "Select library type:",
                ["shared", "static"],
            )
            suffix = "-static" if lib_type == "static" else ""
            path = cmake_dir / f"onnxruntime-linux-arm{suffix}.cmake"
            return parse_url_and_hash(path)

        # x86_64 or aarch64
        options = ["shared", "static"]
        if arch == "x86_64":
            options.append("GPU (CUDA 12, cuDNN 9)")
        elif arch == "aarch64":
            options.append("GPU (multiple CUDA versions)")

        lib_type = select_menu("Select library type:", options)

        if "GPU" in lib_type:
            path = cmake_dir / f"onnxruntime-linux-{arch}-gpu.cmake"
            gpu_deps = parse_url_and_hash(path)
            if not gpu_deps and arch == "aarch64":
                # aarch64 GPU has multiple versions, parse all hashes
                return parse_aarch64_gpu_deps(path)
            return gpu_deps
        elif lib_type == "static":
            path = cmake_dir / f"onnxruntime-linux-{arch}-static.cmake"
        else:
            path = cmake_dir / f"onnxruntime-linux-{arch}.cmake"
        return parse_url_and_hash(path)

    if os_choice == "macOS":
        arch = select_menu(
            "Select architecture:",
            ["arm64", "x86_64", "universal"],
        )
        lib_type = select_menu(
            "Select library type:",
            ["shared", "static"],
        )
        suffix = "-static" if lib_type == "static" else ""
        path = cmake_dir / f"onnxruntime-osx-{arch}{suffix}.cmake"
        return parse_url_and_hash(path)

    if os_choice == "Windows":
        arch = select_menu(
            "Select architecture:",
            ["x64", "x86 (Win32)", "arm64"],
        )
        arch_map = {"x64": "x64", "x86 (Win32)": "x86", "arm64": "arm64"}
        arch_key = arch_map[arch]

        if arch_key == "x64":
            lib_options = ["shared", "static", "GPU", "DirectML"]
        else:
            lib_options = ["shared", "static"]

        lib_type = select_menu("Select library type:", lib_options)

        if lib_type == "DirectML":
            path = cmake_dir / "onnxruntime-win-x64-directml.cmake"
            return parse_url_and_hash(path)
        elif lib_type == "GPU":
            path = cmake_dir / "onnxruntime-win-x64-gpu.cmake"
            return parse_url_and_hash(path)

        suffix = "-static" if lib_type == "static" else ""
        path = cmake_dir / f"onnxruntime-win-{arch_key}{suffix}.cmake"

        if not path.exists():
            print(f"Warning: {path} not found")
            return []

        # Windows has CRT + build type selection
        crt = select_menu(
            "Select MSVC CRT:",
            ["MD (dynamic)", "MT (static)"],
        )
        crt_key = "MD" if "dynamic" in crt else "MT"

        build_type = select_menu(
            "Select build type:",
            ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
        )

        url, sha256 = parse_windows_onnxruntime(path, crt_key, build_type)
        if url and sha256:
            return [("onnxruntime", url, "", sha256)]
        return []

    return []


def parse_aarch64_gpu_deps(path):
    """Parse aarch64 GPU cmake file with multiple version hashes.

    The file has if(v STREQUAL "x.y.z") blocks, each with a hash and
    optionally overridden URLs. The default URL template is:
      https://github.com/csukuangfj/onnxruntime-libs/releases/download/v{v}/onnxruntime-linux-aarch64-gpu-{v}.tar.bz2
    """
    text = path.read_text()
    deps = []

    # Default URL template
    url_template = (
        "https://github.com/csukuangfj/onnxruntime-libs/releases/download"
        "/v{v}/onnxruntime-linux-aarch64-gpu-{v}.tar.bz2"
    )

    # Find all version blocks: if/elseif (v STREQUAL "x.y.z")
    # and extract hash + optional URL overrides within each block
    block_pattern = re.compile(
        r'(?:if|elseif)\(v STREQUAL "([^"]+)"\)\s*\n(.*?)(?=\belseif\b|\belse\b|\bendif\b)',
        re.DOTALL,
    )

    for m in block_pattern.finditer(text):
        version = m.group(1)
        block = m.group(2)

        # Extract hash from this block
        hash_m = re.search(r'set\(onnxruntime_HASH\s+"SHA256=([^"]+)"\)', block)
        if not hash_m:
            continue
        sha256 = hash_m.group(1)

        # Check for URL override in this block
        url_m = re.search(r'set\(onnxruntime_URL\s+"(https://[^"]+)"\)', block)
        url2_m = re.search(r'set\(onnxruntime_URL2\s+"(https://[^"]+)"\)', block)

        if url_m:
            url = url_m.group(1)
            url2 = url2_m.group(1) if url2_m else ""
        else:
            # Use template
            url = url_template.format(v=version)
            url2 = ""

        deps.append(("onnxruntime", url, url2, sha256))

    return deps


_canonical_filenames = None


def build_canonical_filename_lookup(cmake_dir):
    """Build a mapping from URL basename to canonical local filename.

    Scans cmake files for $ENV{HOME}/Downloads/<filename> entries, which
    define the expected local filenames (e.g., "cargs-1.0.3.tar.gz" even
    when the URL basename is just "v1.0.3.tar.gz").
    """
    lookup = {}
    dl_pattern = re.compile(
        r'Downloads/([^\s"]+\.(?:tar\.(?:gz|bz2)|zip|tgz|nupkg))'
    )
    for cmake_file in sorted(cmake_dir.glob("*.cmake")):
        text = cmake_file.read_text(errors="ignore")
        for m in dl_pattern.finditer(text):
            local_name = m.group(1)
            # Also index by stripped version (without project prefix)
            # so URLs like "v1.0.3.tar.gz" can map to "cargs-1.0.3.tar.gz"
            lookup[local_name] = local_name
    return lookup


def filename_from_url(url, dep_name="", cmake_dir=None):
    """Get a descriptive filename for a download URL.

    Uses canonical filenames from cmake files when available.
    Falls back to URL basename with dep_name prefix.
    """
    global _canonical_filenames
    basename = os.path.basename(urllib.parse.urlparse(url).path)
    if not basename:
        basename = "unknown"

    # Build lookup on first call
    if cmake_dir and _canonical_filenames is None:
        _canonical_filenames = build_canonical_filename_lookup(cmake_dir)

    # Try to find a matching canonical name by stripping version/hash prefixes
    if _canonical_filenames and dep_name:
        # Normalize dep_name: underscores → hyphens
        norm_name = dep_name.replace("_", "-")

        # Check all canonical names that start with this dep name
        for canonical in _canonical_filenames.values():
            # Match: canonical starts with dep_name and contains the same
            # version/hash portion from the URL basename
            if canonical.startswith(norm_name + "-"):
                # Extract the suffix after "{name}-"
                suffix = canonical[len(norm_name) + 1:]
                # Check if basename contains the suffix or vice versa
                # (handles "v1.0.3.tar.gz" ↔ "1.0.3.tar.gz")
                base_no_v = basename.lstrip("v")
                suf_no_v = suffix.lstrip("v")
                if base_no_v == suf_no_v or basename == suffix:
                    return canonical

    # If basename is generic (version-only or hash-only), prefix with dep name
    if dep_name and re.match(r'^(v\d|[0-9a-f]{8,})', basename):
        return f"{dep_name.replace('_', '-')}-{basename.lstrip('v')}"

    return basename


def download_file(url, filepath, expected_sha256, retries=3):
    """Download a file and verify its SHA256 hash.

    Returns True if successful.
    """
    if filepath.exists():
        # Verify existing file
        sha256 = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if sha256 == expected_sha256:
            print(f"  Already exists with correct hash: {filepath.name}")
            return True
        else:
            print(f"  Existing file has wrong hash, re-downloading: {filepath.name}")
            filepath.unlink()

    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  Retry {attempt}/{retries}: {url}")
        else:
            print(f"  Downloading: {url}")

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            if attempt < retries:
                continue
            return False

        # Verify hash
        sha256 = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if sha256 != expected_sha256:
            print(f"  ERROR: SHA256 mismatch!")
            print(f"    Expected: {expected_sha256}")
            print(f"    Got:      {sha256}")
            filepath.unlink()
            if attempt < retries:
                continue
            return False

        print(f"  OK: {filepath.name}")
        return True

    return False


def main():
    cmake_dir = get_cmake_dir()
    print(f"Scanning cmake files in: {cmake_dir}")

    # Discover common deps
    common_deps = get_hardcoded_deps() + discover_common_deps(cmake_dir)
    print(f"\nFound {len(common_deps)} common (platform-independent) dependencies:")
    for name, url, url2, sha256 in common_deps:
        print(f"  {name}: {filename_from_url(url, name, cmake_dir)}")

    # Ask about onnxruntime
    ort_deps = []
    if ask_yes_no("\nDownload onnxruntime (platform-specific)?"):
        ort_deps = select_onnxruntime_deps(cmake_dir)
        if ort_deps:
            print(f"\nSelected {len(ort_deps)} onnxruntime file(s):")
            for name, url, url2, sha256 in ort_deps:
                print(f"  {name}: {filename_from_url(url, name, cmake_dir)}")
    else:
        print("\nSkipping onnxruntime download.")

    all_deps = common_deps + ort_deps
    if not all_deps:
        print("No dependencies selected.")
        return

    # Ask for download directory
    print()
    default_dir = str(Path.home() / "Downloads")
    try:
        dl_dir = input(
            f"Download directory [{default_dir}]: "
        ).strip() or default_dir
    except EOFError:
        dl_dir = default_dir

    dl_path = Path(dl_dir)
    dl_path.mkdir(parents=True, exist_ok=True)

    # Download all
    print(f"\nDownloading {len(all_deps)} file(s) to {dl_path}...")
    downloaded = []
    failed = []
    for name, url, url2, sha256 in all_deps:
        fname = filename_from_url(url, name, cmake_dir)
        filepath = dl_path / fname
        if download_file(url, filepath, sha256):
            downloaded.append(fname)
        else:
            # Try mirror URL
            if url2:
                print(f"  Trying mirror: {url2}")
                if download_file(url2, filepath, sha256):
                    downloaded.append(fname)
                    continue
            failed.append(fname)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download complete: {len(downloaded)} succeeded, {len(failed)} failed.")
    if downloaded:
        print(f"\nDownloaded ({len(downloaded)}):")
        for f in downloaded:
            print(f"  {f}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
