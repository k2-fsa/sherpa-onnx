# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

configure_file(
  "${CMAKE_ROOT}/Modules/CMakeRustCompiler.cmake.in"
  "${CMAKE_PLATFORM_INFO_DIR}/CMakeRustCompiler.cmake"
  @ONLY)

include(${CMAKE_PLATFORM_INFO_DIR}/CMakeRustCompiler.cmake)
