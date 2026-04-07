# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Helper functions for chipStar (spirv platform) HIP support

# Set chipStar compile flags in CMAKE_HIP_COMPILE_OBJECT
function(_cmake_chipstar_set_compiler_flags)
  if(NOT CMAKE_HIP_PLATFORM STREQUAL "spirv" OR NOT CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    return()
  endif()
  set(_hip_path "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
  set(_chipstar_flags "--offload=spirv64 --hip-path=\"${_hip_path}\" -nogpulib -nohipwrapperinc -include \"${_hip_path}/include/hip/spirv_fixups.h\"")
  set(CMAKE_HIP_COMPILE_OBJECT
    "<CMAKE_HIP_COMPILER> ${_chipstar_flags} <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -x hip -c <SOURCE>" PARENT_SCOPE)
endfunction()

# Set link commands for chipStar to include libCHIP.so
# This makes CMake treat it as if the compiler driver links it implicitly
function(_cmake_chipstar_set_link_commands)
  if(NOT CMAKE_HIP_PLATFORM STREQUAL "spirv")
    return()
  endif()
  set(_hip_path "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
  set(_chipstar_lib "${_hip_path}/lib/libCHIP.so")
  # Set link command templates with libCHIP.so appended (quoted for paths with spaces)
  set(CMAKE_HIP_LINK_EXECUTABLE
    "<CMAKE_HIP_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES> \"${_chipstar_lib}\"" PARENT_SCOPE)
  set(CMAKE_HIP_CREATE_SHARED_LIBRARY
    "<CMAKE_HIP_COMPILER> <CMAKE_SHARED_LIBRARY_HIP_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> \"${_chipstar_lib}\"" PARENT_SCOPE)
  set(CMAKE_HIP_CREATE_SHARED_MODULE
    "<CMAKE_HIP_COMPILER> <CMAKE_SHARED_LIBRARY_HIP_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> \"${_chipstar_lib}\"" PARENT_SCOPE)
endfunction()



# Set compiler ID test flags for chipStar (called before CMAKE_HIP_COMPILER_ROCM_ROOT is set)
function(_cmake_chipstar_set_compiler_id_flags)
  if(NOT CMAKE_HIP_PLATFORM STREQUAL "spirv")
    return()
  endif()
  # At this point CMAKE_HIP_COMPILER_ROCM_ROOT isn't set yet, use HIP_PATH or hipconfig
  set(_hip_path "$ENV{HIP_PATH}")
  if(NOT IS_DIRECTORY "${_hip_path}")
    execute_process(COMMAND hipconfig --rocmpath
      OUTPUT_VARIABLE _hip_path OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(IS_DIRECTORY "${_hip_path}")
    file(TO_CMAKE_PATH "${_hip_path}" _hip_path)
    # All flags in one string so they're used together. -no-hip-rt prevents linking amdhip64.
    list(APPEND CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST
      "-v --offload=spirv64 --hip-path=\"${_hip_path}\" -nogpulib -nohipwrapperinc -no-hip-rt \"${_hip_path}/lib/libCHIP.so\" -include \"${_hip_path}/include/hip/spirv_fixups.h\"")
    set(CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST "${CMAKE_HIP_COMPILER_ID_TEST_FLAGS_FIRST}" PARENT_SCOPE)
  endif()
endfunction()
