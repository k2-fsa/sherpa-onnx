# Copyright     2020 Fangjun Kuang (csukuangfj@gmail.com)
# See ../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(download_googltest)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    # FetchContent is available since 3.11,
    # we've copied it to ${CMAKE_SOURCE_DIR}/cmake/Modules
    # so that it can be used in lower CMake versions.
    message(STATUS "Use FetchContent provided by sherpa-onnx")
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(googletest_URL  "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
  set(googletest_HASH "SHA256=9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb")

  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_Declare(googletest
    URL               ${googletest_URL}
    URL_HASH          ${googletest_HASH}
  )

  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    message(STATUS "Downloading googletest")
    FetchContent_Populate(googletest)
  endif()
  message(STATUS "googletest is downloaded to ${googletest_SOURCE_DIR}")
  message(STATUS "googletest's binary dir is ${googletest_BINARY_DIR}")

  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON) # to solve the following warning on macOS
  endif()
  #[==[
  -- Generating done
    Policy CMP0042 is not set: MACOSX_RPATH is enabled by default.  Run "cmake
    --help-policy CMP0042" for policy details.  Use the cmake_policy command to
    set the policy and suppress this warning.

    MACOSX_RPATH is not specified for the following targets:

      gmock
      gmock_main
      gtest
      gtest_main

  This warning is for project developers.  Use -Wno-dev to suppress it.
  ]==]

  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(gtest
    INTERFACE
      ${googletest_SOURCE_DIR}/googletest/include
      ${googletest_SOURCE_DIR}/googlemock/include
  )
endfunction()

download_googltest()
