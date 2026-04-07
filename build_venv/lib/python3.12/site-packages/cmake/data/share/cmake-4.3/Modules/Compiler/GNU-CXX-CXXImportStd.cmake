function (_cmake_cxx_find_modules_json)
  if (NOT CMAKE_CXX_STANDARD_LIBRARY STREQUAL "libstdc++")
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "Only `libstdc++` is supported" PARENT_SCOPE)
    return ()
  endif ()

  if (NOT CMAKE_CXX_STDLIB_MODULES_JSON)
    execute_process(
      COMMAND
        "${CMAKE_CXX_COMPILER}"
        ${CMAKE_CXX_COMPILER_ID_ARG1}
        -print-file-name=libstdc++.modules.json
      OUTPUT_VARIABLE _gnu_libstdcxx_modules_json_file
      ERROR_VARIABLE _gnu_libstdcxx_modules_json_file_err
      RESULT_VARIABLE _gnu_libstdcxx_modules_json_file_res
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE)
    if (_gnu_libstdcxx_modules_json_file_res)
      set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "Could not find `libstdc++.modules.json` resource" PARENT_SCOPE)
      return ()
    endif ()
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${_gnu_libstdcxx_modules_json_file}" PARENT_SCOPE)
  endif ()
endfunction ()
