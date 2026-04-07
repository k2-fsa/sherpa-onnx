# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

if(NOT "${CMAKE_GENERATOR}" MATCHES "^Ninja")
  message(FATAL_ERROR "Rust language not supported by \"${CMAKE_GENERATOR}\" generator")
endif()

set(CMAKE_Rust_COMPILER_INIT "rustc")
set(CMAKE_Rust_COMPILER_HINTS "$ENV{HOME}/.cargo/bin")

_cmake_find_compiler(Rust)

get_filename_component(RUSTC_REAL "${CMAKE_Rust_COMPILER}" REALPATH)
get_filename_component(RUSTC_FILENAME "${RUSTC_REAL}" NAME)

# When rustup is used for installing rust, rustc will just be a symlink to rustup. In such cases,
# we need to query rustup for underlying rustc path.
if(RUSTC_FILENAME STREQUAL "rustup")
  get_filename_component(RUSTC_DIR "${CMAKE_Rust_COMPILER}" DIRECTORY)
  set(RUSTUP_PATH "${RUSTC_DIR}/rustup")

  # Fix RUSTUP_HOME in ctest.
  if(RUSTC_FILENAME STREQUAL "rustup" AND NOT "$ENV{CTEST_REAL_HOME}" STREQUAL "" AND "$ENV{RUSTUP_HOME}" STREQUAL "")
    set(ENV{RUSTUP_HOME} "$ENV{CTEST_REAL_HOME}/.rustup")
  endif()

  execute_process(
    COMMAND ${RUSTUP_PATH} which rustc
    OUTPUT_VARIABLE REAL_RUSTC
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if("${REAL_RUSTC}" STREQUAL "")
    message(FATAL_ERROR "Failed to find path to real rustc")
  endif()

  set_property(CACHE CMAKE_Rust_COMPILER PROPERTY VALUE "${REAL_RUSTC}")
endif()

if(CMAKE_Rust_COMPILER)
  set(CMAKE_Rust_COMPILER_WORKS TRUE)
endif()

configure_file(
  "${CMAKE_ROOT}/Modules/CMakeRustCompiler.cmake.in"
  "${CMAKE_PLATFORM_INFO_DIR}/CMakeRustCompiler.cmake"
  @ONLY)
