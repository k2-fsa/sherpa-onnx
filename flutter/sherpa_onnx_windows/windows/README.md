# Introduction

`*.dll` files are generated dynamically using GitHub actions during a new release.

We don't check-in pre-built library files into git.

The published package carries two sets of libraries, and `CMakeLists.txt`
picks the one matching the architecture a Flutter app is built for:

- `windows/x64/*.dll` for x64
- `windows/arm64/*.dll` for arm64 (Windows on ARM)
