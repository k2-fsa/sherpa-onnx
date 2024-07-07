# tts


# Fix for Linux

If you get the following errors on Linux,

```
Building Linux application...
CMake Error at /usr/local/share/cmake-3.29/Modules/FindPkgConfig.cmake:634 (message):
  The following required packages were not found:

   - gstreamer-1.0

Call Stack (most recent call first):
  /usr/local/share/cmake-3.29/Modules/FindPkgConfig.cmake:862 (_pkg_check_modules_internal)
  flutter/ephemeral/.plugin_symlinks/audioplayers_linux/linux/CMakeLists.txt:24 (pkg_check_modules)
```

please run:

```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

See also <https://github.com/bluefireteam/audioplayers/tree/main/packages/audioplayers_linux#setup-for-linux>
for the above error.
