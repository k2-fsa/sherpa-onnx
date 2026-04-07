function (_cmake_cxx_find_modules_json)
  if (CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libc++")
    set(_clang_modules_json_impl "libc++")
  elseif (CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libstdc++")
    set(_clang_modules_json_impl "libstdc++")
  else ()
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "Only `libc++` and `libstdc++` are supported" PARENT_SCOPE)
    return ()
  endif ()

  if (NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    execute_process(
      COMMAND
        "${CMAKE_CXX_COMPILER}"
        ${CMAKE_CXX_COMPILER_ID_ARG1}
        "-print-file-name=${_clang_modules_json_impl}.modules.json"
      OUTPUT_VARIABLE _clang_libcxx_modules_json_file
      ERROR_VARIABLE _clang_libcxx_modules_json_file_err
      RESULT_VARIABLE _clang_libcxx_modules_json_file_res
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    if (_clang_libcxx_modules_json_file_res)
      set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "Could not find `${_clang_modules_json_impl}.modules.json` resource" PARENT_SCOPE)
      return ()
    endif ()
    cmake_path(CONVERT "${_clang_libcxx_modules_json_file}" TO_CMAKE_PATH_LIST CMAKE_CXX_STDLIB_MODULES_JSON NORMALIZE)
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${CMAKE_CXX_STDLIB_MODULES_JSON}" PARENT_SCOPE)
  endif ()

  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "18.1.2" AND
      CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libc++")
    # The original PR had a key spelling mismatch internally. Do not support it
    # and instead require a release known to have the fix.
    # https://github.com/llvm/llvm-project/pull/83036
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "LLVM 18.1.2 is required for `${_clang_modules_json_impl}.modules.json` format fix" PARENT_SCOPE)
    return ()
  endif ()
endfunction ()
