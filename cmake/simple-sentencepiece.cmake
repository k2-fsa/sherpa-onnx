function(download_simple_sentencepiece)
  include(FetchContent)

  set(simple-sentencepiece_URL  "https://github.com/pkufool/simple-sentencepiece/archive/refs/tags/v0.7.tar.gz")
  set(simple-sentencepiece_URL2 "https://hf-mirror.com/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/simple-sentencepiece-0.7.tar.gz")
  set(simple-sentencepiece_HASH "SHA256=1748a822060a35baa9f6609f84efc8eb54dc0e74b9ece3d82367b7119fdc75af")

  # If you don't have access to the Internet,
  # please pre-download simple-sentencepiece
  set(possible_file_locations
    $ENV{HOME}/Downloads/simple-sentencepiece-0.7.tar.gz
    ${CMAKE_SOURCE_DIR}/simple-sentencepiece-0.7.tar.gz
    ${CMAKE_BINARY_DIR}/simple-sentencepiece-0.7.tar.gz
    /tmp/simple-sentencepiece-0.7.tar.gz
    /star-fj/fangjun/download/github/simple-sentencepiece-0.7.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(simple-sentencepiece_URL  "${f}")
      file(TO_CMAKE_PATH "${simple-sentencepiece_URL}" simple-sentencepiece_URL)
      message(STATUS "Found local downloaded simple-sentencepiece: ${simple-sentencepiece_URL}")
      set(simple-sentencepiece_URL2)
      break()
    endif()
  endforeach()

  set(SBPE_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SBPE_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(simple-sentencepiece
    URL
      ${simple-sentencepiece_URL}
      ${simple-sentencepiece_URL2}
    URL_HASH
      ${simple-sentencepiece_HASH}
  )

  FetchContent_GetProperties(simple-sentencepiece)
  if(NOT simple-sentencepiece_POPULATED)
    message(STATUS "Downloading simple-sentencepiece ${simple-sentencepiece_URL}")
    FetchContent_Populate(simple-sentencepiece)
  endif()
  message(STATUS "simple-sentencepiece is downloaded to ${simple-sentencepiece_SOURCE_DIR}")

  # Patch ssentencepiece to disable threading on WASM (std::thread not available).
  if(SHERPA_ONNX_ENABLE_WASM)
    # Replace threadpool.h with a WASM-safe stub that provides a no-op ThreadPool.
    set(_tp_header "${simple-sentencepiece_SOURCE_DIR}/ssentencepiece/csrc/threadpool.h")
    file(WRITE "${_tp_header}" [=[
// WASM-safe ThreadPool stub (std::thread not available).
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

class ThreadPool {
 public:
  ThreadPool(size_t) {}
  template<class F, class... Args>
  std::future<std::invoke_result_t<F, Args...>>
  enqueue(F&& f, Args&&... args) {
    // Run synchronously — no threading in WASM.
    using return_type = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    (*task)();
    return res;
  }
};

#endif  // THREAD_POOL_H
]=])
    message(STATUS "Patched ssentencepiece for WASM (ThreadPool stub installed)")
  endif()

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${simple-sentencepiece_SOURCE_DIR} ${simple-sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)

  if(TARGET ssentencepiece_core AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    target_compile_options(ssentencepiece_core PRIVATE -Wno-deprecated-declarations)
  endif()

  if(_build_shared_libs_bak)
    set_target_properties(ssentencepiece_core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(ssentencepiece_core
    PUBLIC
      ${simple-sentencepiece_SOURCE_DIR}/
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS ssentencepiece_core DESTINATION lib)
  endif()
endfunction()

download_simple_sentencepiece()
