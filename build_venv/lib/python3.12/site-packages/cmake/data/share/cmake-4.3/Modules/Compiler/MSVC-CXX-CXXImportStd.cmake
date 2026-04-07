function (_cmake_cxx_find_modules_json)
  if (CMAKE_CXX_STDLIB_MODULES_JSON)
    return ()
  endif ()

  find_file(_msvc_modules_json_file
    NAME modules.json
    HINTS
      "$ENV{VCToolsInstallDir}/modules"
    PATHS
      "$ENV{INCLUDE}"
      "${CMAKE_CXX_COMPILER}/../../.."
      "${CMAKE_CXX_COMPILER}/../.."    # msvc-wine layout
    PATH_SUFFIXES
      ../modules
    NO_CACHE)

  # Without this file, we do not have modules installed.
  if (NOT EXISTS "${_msvc_modules_json_file}")
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "Could not find `modules.json` resource" PARENT_SCOPE)
    return ()
  endif ()

  file(READ "${_msvc_modules_json_file}" _msvc_modules_json)
  string(JSON _msvc_json_modules ERROR_VARIABLE _msvc_json_err GET "${_msvc_modules_json}" "modules")

  # This is probably a conforming module metadata file, use it as is
  if (_msvc_json_modules)
    set(CMAKE_CXX_STDLIB_MODULES_JSON "${_msvc_modules_json_file}" PARENT_SCOPE)
    return ()
  endif ()

  # Otherwise it's a Microsoft STL-style modules.json, check if we recognize it
  string(JSON _msvc_json_version GET "${_msvc_modules_json}" "version")
  string(JSON _msvc_json_revision GET "${_msvc_modules_json}" "revision")
  # Require version 1.
  if (NOT _msvc_json_version EQUAL "1")
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "`modules.json` version ${_msvc_json_version}.${_msvc_json_revision} is not recognized" PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _msvc_json_library GET "${_msvc_modules_json}" "library")
  # Bail if we don't understand the library.
  if (NOT _msvc_json_library STREQUAL "microsoft/STL")
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "`modules.json` library `${_msvc_json_library}` is not recognized" PARENT_SCOPE)
    return ()
  endif ()

  string(JSON _msvc_json_sources_len LENGTH "${_msvc_modules_json}" "module-sources")
  # Bail if there aren't any sources
  if (NOT _msvc_json_sources_len)
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "`modules.json` does not list any available sources" PARENT_SCOPE)
    return ()
  endif ()

  set(_msvc_module_metadata [=[{
    "version": 1,
    "revision": 1,
    "modules": []
  }]=])

  cmake_path(GET _msvc_modules_json_file PARENT_PATH _msvc_base_dir)
  math(EXPR _msvc_json_sources_len "${_msvc_json_sources_len}-1")
  foreach (idx RANGE ${_msvc_json_sources_len})
    string(JSON _msvc_source GET "${_msvc_modules_json}" "module-sources" ${idx})

    set(_msvc_path ${_msvc_source})
    cmake_path(IS_RELATIVE _msvc_path _msvc_is_rel)
    if (_msvc_is_rel)
      cmake_path(ABSOLUTE_PATH _msvc_path BASE_DIRECTORY "${_msvc_base_dir}")
    endif ()

    if (_msvc_source MATCHES "std.ixx")
      string(JSON _msvc_module_metadata
        SET "${_msvc_module_metadata}" "modules" ${idx}
        "{
            \"logical-name\": \"std\",
            \"source-path\": \"${_msvc_path}\",
            \"is-std-library\": true
        }"
      )
    elseif (_msvc_source MATCHES "std.compat.ixx")
      string(JSON _msvc_module_metadata
        SET "${_msvc_module_metadata}" "modules" ${idx}
        "{
            \"logical-name\": \"std.compat\",
            \"source-path\": \"${_msvc_path}\",
            \"is-std-library\": true
        }"
      )
    endif ()
  endforeach()

  string(JSON _msvc_module_metadata_modules_len LENGTH ${_msvc_module_metadata} "modules")

  # Bail if we didn't recognize any of the modules
  if (NOT _msvc_module_metadata_modules_len)
    set(CMAKE_CXX_COMPILER_IMPORT_STD_ERROR_MESSAGE "`modules.json` did not contain any recognized sources (std.ixx, std.compat.ixx)" PARENT_SCOPE)
    return ()
  endif ()

  file(WRITE "${CMAKE_PLATFORM_INFO_DIR}/msvcstl.modules.json" "${_msvc_module_metadata}")
  set(CMAKE_CXX_STDLIB_MODULES_JSON "${CMAKE_PLATFORM_INFO_DIR}/msvcstl.modules.json" PARENT_SCOPE)
endfunction ()
