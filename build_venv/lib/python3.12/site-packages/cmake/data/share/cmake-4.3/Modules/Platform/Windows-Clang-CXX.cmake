include(Platform/Windows-Clang)
set(_COMPILE_CXX_MSVC " -TP")
__windows_compiler_clang(CXX)

if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
    AND CMAKE_DEPFILE_FLAGS_CXX)
  # dependencies are computed by the compiler itself
  set(CMAKE_CXX_DEPFILE_FORMAT gcc)
  set(CMAKE_CXX_DEPENDS_USE_COMPILER TRUE)
endif()
