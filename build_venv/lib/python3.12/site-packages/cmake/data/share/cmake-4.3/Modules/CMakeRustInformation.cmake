# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(CMakeLanguageInformation)

if(UNIX)
  set(CMAKE_Rust_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Rust_OUTPUT_EXTENSION .obj)
endif()

set(CMAKE_Rust_LIBRARY_PATH_FLAG "-L ")
set(CMAKE_Rust_LINK_LIBRARY_FILE_FLAG "-C link-arg=")
set(CMAKE_EXECUTABLE_RUNTIME_Rust_FLAG "-C link-arg=-Wl,-rpath,")
set(CMAKE_EXECUTABLE_RUNTIME_Rust_FLAG_SEP ",")

set(CMAKE_Rust_FLAGS_DEBUG_INIT "-C opt-level=0 -g")
set(CMAKE_Rust_FLAGS_RELEASE_INIT "-O")
set(CMAKE_Rust_FLAGS_RELWITHDEBINFO_INIT "-O -g")
set(CMAKE_Rust_FLAGS_MINSIZEREL_INIT "-C opt-level=z")

cmake_initialize_per_config_variable(CMAKE_Rust_FLAGS "Flags used by the Rust compiler")

if(NOT CMAKE_Rust_CREATE_STATIC_LIBRARY)
  set(CMAKE_Rust_CREATE_STATIC_LIBRARY "${CMAKE_Rust_COMPILER} <LANGUAGE_COMPILE_FLAGS> --crate-type=staticlib <RUST_SOURCES> -o <TARGET> -C link-args=\"<RUST_OBJECT_DEPS>\"")
endif()

if(NOT CMAKE_Rust_CREATE_SHARED_LIBRARY)
  set(CMAKE_Rust_CREATE_SHARED_LIBRARY "${CMAKE_Rust_COMPILER} <LANGUAGE_COMPILE_FLAGS> --crate-type=cdylib <RUST_SOURCES> -o <TARGET> <LINK_FLAGS> <LINK_LIBRARIES> -C link-args=\"<RUST_OBJECT_DEPS>\"")
endif()

# Deadcode warnings are not useful when generating object files.
if(NOT CMAKE_Rust_COMPILE_OBJECT)
  set(CMAKE_Rust_COMPILE_OBJECT "${CMAKE_Rust_COMPILER} <FLAGS> -A dead_code --crate-type=lib --emit=obj=<OBJECT>,dep-info=<DEP_FILE> <SOURCE>")
endif()

if(NOT CMAKE_Rust_LINK_EXECUTABLE)
  set(CMAKE_Rust_LINK_EXECUTABLE "${CMAKE_Rust_COMPILER} <FLAGS> --crate-type=bin <RUST_SOURCES> -o <TARGET> <LINK_FLAGS> <LINK_LIBRARIES> -C link-args=\"<RUST_OBJECT_DEPS>\"")
endif()

set(CMAKE_Rust_INFORMATION_LOADED 1)
