include(Platform/Windows-Clang)
set(_COMPILE_HIP_MSVC " -x hip")
__windows_compiler_clang(HIP)

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_HIP)
  # dependencies are computed by the compiler itself
  set(CMAKE_HIP_DEPFILE_FORMAT gcc)
  set(CMAKE_HIP_DEPENDS_USE_COMPILER TRUE)
endif()
