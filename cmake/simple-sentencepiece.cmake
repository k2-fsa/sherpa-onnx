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

class ThreadPool {
 public:
  ThreadPool(size_t) {}
  // C++14 compatible: use auto return type with decltype.
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<decltype(f(args...))> {
    // Run synchronously — no threading in WASM.
    using return_type = decltype(f(args...));
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
  elseif(CMAKE_SYSTEM_NAME STREQUAL Android)
    # On Termux, the static onnxruntime lib references __cxa_init_primary_exception
    # which is missing from libc++abi. Remove all threading from ssentencepiece:
    # no std::thread, std::future, std::promise, or ThreadPool.

    # 1. Replace threadpool.h with an empty stub.
    set(_tp_header "${simple-sentencepiece_SOURCE_DIR}/ssentencepiece/csrc/threadpool.h")
    file(WRITE "${_tp_header}" [=[
// Empty stub — threading removed for Android/Termux.
#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#endif  // THREAD_POOL_H
]=])

    # 2. Patch ssentencepiece.h: remove ThreadPool include, thread usage, and pool_ member.
    set(_sph "${simple-sentencepiece_SOURCE_DIR}/ssentencepiece/csrc/ssentencepiece.h")
    file(READ "${_sph}" _sph_content)
    string(REPLACE "#include \"ssentencepiece/csrc/threadpool.h\"\n" "" _sph_content "${_sph_content}")
    string(REPLACE "int32_t num_threads = std::thread::hardware_concurrency()" "int32_t num_threads = 1" _sph_content "${_sph_content}")
    string(REPLACE "pool_ = std::make_unique<ThreadPool>(num_threads);\n    Build(is);" "Build(is);" _sph_content "${_sph_content}")
    string(REPLACE "pool_ = std::make_unique<ThreadPool>(num_threads);\n    Build(vocab_path);" "Build(vocab_path);" _sph_content "${_sph_content}")
    string(REPLACE "pool_ = std::make_unique<ThreadPool>(num_threads);" "" _sph_content "${_sph_content}")
    string(REPLACE "std::unique_ptr<ThreadPool> pool_;" "" _sph_content "${_sph_content}")
    file(WRITE "${_sph}" "${_sph_content}")

    # 3. Patch ssentencepiece.cc: replace threaded Encode/Decode with synchronous loops.
    set(_spc "${simple-sentencepiece_SOURCE_DIR}/ssentencepiece/csrc/ssentencepiece.cc")
    file(READ "${_spc}" _spc_content)

    # Replace threaded Encode (vector<string> version)
    string(REPLACE
      [==[  ostrs->resize(strs.size());
  std::vector<std::future<void>> results;
  for (int32_t i = 0; i < strs.size(); ++i) {
    results.emplace_back(pool_->enqueue([this, i, &strs, ostrs] {
      return this->Encode(strs[i], &((*ostrs)[i]));
    }));
  }

  for (auto &&result : results) {
    result.get();
  }]==]
      [==[  ostrs->resize(strs.size());
  for (int32_t i = 0; i < strs.size(); ++i) {
    this->Encode(strs[i], &((*ostrs)[i]));
  }]==]
      _spc_content "${_spc_content}")

    # Replace threaded Encode (vector<int32_t> version)
    string(REPLACE
      [==[  oids->resize(strs.size());
  std::vector<std::future<void>> results;
  for (int32_t i = 0; i < strs.size(); ++i) {
    results.emplace_back(pool_->enqueue([this, i, &strs, oids] {
      return this->Encode(strs[i], &((*oids)[i]));
    }));
  }

  for (auto &&result : results) {
    result.get();
  }]==]
      [==[  oids->resize(strs.size());
  for (int32_t i = 0; i < strs.size(); ++i) {
    this->Encode(strs[i], &((*oids)[i]));
  }]==]
      _spc_content "${_spc_content}")

    # Replace threaded Decode (vector version)
    string(REPLACE
      [==[  std::vector<std::string> res;
  std::vector<std::future<std::string>> results;
  for (const auto &id : ids) {
    results.emplace_back(
        pool_->enqueue([this, &id] { return this->Decode(id); }));
  }
  for (auto &&result : results) {
    res.push_back(result.get());
  }
  return res;]==]
      [==[  std::vector<std::string> res;
  for (const auto &id : ids) {
    res.push_back(this->Decode(id));
  }
  return res;]==]
      _spc_content "${_spc_content}")

    file(WRITE "${_spc}" "${_spc_content}")

    message(STATUS "Patched ssentencepiece for Android: all threading removed")
  endif()

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  # Skip the C++14 compiler check for WASM (Emscripten supports it but the
  # CMake test may fail depending on flags).
  if(SHERPA_ONNX_ENABLE_WASM)
    set(SBPE_COMPILER_SUPPORTS_CXX14 ON CACHE BOOL "" FORCE)
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
