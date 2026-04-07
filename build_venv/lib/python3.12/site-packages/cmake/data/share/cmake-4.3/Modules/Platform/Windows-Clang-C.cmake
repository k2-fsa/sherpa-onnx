include(Platform/Windows-Clang)
__windows_compiler_clang(C)

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_C)
  # dependencies are computed by the compiler itself
  set(CMAKE_C_DEPFILE_FORMAT gcc)
  set(CMAKE_C_DEPENDS_USE_COMPILER TRUE)
endif()
