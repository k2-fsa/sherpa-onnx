include_guard(GLOBAL)

function(download_utf8proc)
  include(FetchContent)

  # utf8proc owns all Unicode handling here: NFC/NFD normalization, simple
  # case mapping and general categories. Keeping that in a library that tracks
  # the Unicode standard beats maintaining our own case/composition tables.
  set(utf8proc_URL   "https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.12.0.tar.gz")
  set(utf8proc_HASH  "SHA256=f564011d38b2888d583d510b08e69ffa15aa117155db1b9b49ef1dfe1fa25111")

  set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
  set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download utf8proc
  set(possible_file_locations
    $ENV{HOME}/Downloads/utf8proc-2.12.0.tar.gz
    ${CMAKE_SOURCE_DIR}/utf8proc-2.12.0.tar.gz
    ${CMAKE_BINARY_DIR}/utf8proc-2.12.0.tar.gz
    /tmp/utf8proc-2.12.0.tar.gz
    /star-fj/fangjun/download/github/utf8proc-2.12.0.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(utf8proc_URL "${f}")
      file(TO_CMAKE_PATH "${utf8proc_URL}" utf8proc_URL)
      message(STATUS "Found local downloaded utf8proc: ${utf8proc_URL}")
      break()
    endif()
  endforeach()

  FetchContent_Declare(utf8proc
    URL
      ${utf8proc_URL}
    URL_HASH          ${utf8proc_HASH}
  )

  FetchContent_GetProperties(utf8proc)
  if(NOT utf8proc_POPULATED)
    message(STATUS "Downloading utf8proc from ${utf8proc_URL}")
    FetchContent_Populate(utf8proc)
  endif()
  message(STATUS "utf8proc is downloaded to ${utf8proc_SOURCE_DIR}")

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  add_subdirectory(${utf8proc_SOURCE_DIR} ${utf8proc_BINARY_DIR} EXCLUDE_FROM_ALL)

  # utf8proc's C symbols (utf8proc_map, utf8proc_free, ...) are generic enough
  # that the host application may link its own copy of utf8proc (text libraries
  # commonly bundle it). Merging two copies makes the linker bind our calls to
  # the foreign one; if the two versions disagree (e.g. utf8proc_property_t
  # layout), that corrupts memory at runtime — the same failure mode as the
  # kissfft clash in https://github.com/k2-fsa/sherpa-onnx/issues/2588.
  # Rename every public C symbol of utf8proc with a sherpa_onnx_ prefix. PUBLIC
  # is required so our own sources see the renamed declarations.
  target_compile_definitions(utf8proc PUBLIC
    utf8proc_category=sherpa_onnx_utf8proc_category
    utf8proc_category_string=sherpa_onnx_utf8proc_category_string
    utf8proc_charwidth=sherpa_onnx_utf8proc_charwidth
    utf8proc_charwidth_ambiguous=sherpa_onnx_utf8proc_charwidth_ambiguous
    utf8proc_codepoint_valid=sherpa_onnx_utf8proc_codepoint_valid
    utf8proc_decompose=sherpa_onnx_utf8proc_decompose
    utf8proc_decompose_char=sherpa_onnx_utf8proc_decompose_char
    utf8proc_decompose_custom=sherpa_onnx_utf8proc_decompose_custom
    utf8proc_encode_char=sherpa_onnx_utf8proc_encode_char
    utf8proc_errmsg=sherpa_onnx_utf8proc_errmsg
    utf8proc_free=sherpa_onnx_utf8proc_free
    utf8proc_get_property=sherpa_onnx_utf8proc_get_property
    utf8proc_grapheme_break=sherpa_onnx_utf8proc_grapheme_break
    utf8proc_grapheme_break_stateful=sherpa_onnx_utf8proc_grapheme_break_stateful
    utf8proc_islower=sherpa_onnx_utf8proc_islower
    utf8proc_isupper=sherpa_onnx_utf8proc_isupper
    utf8proc_iterate=sherpa_onnx_utf8proc_iterate
    utf8proc_map=sherpa_onnx_utf8proc_map
    utf8proc_map_custom=sherpa_onnx_utf8proc_map_custom
    utf8proc_NFC=sherpa_onnx_utf8proc_NFC
    utf8proc_NFD=sherpa_onnx_utf8proc_NFD
    utf8proc_NFKC=sherpa_onnx_utf8proc_NFKC
    utf8proc_NFKC_Casefold=sherpa_onnx_utf8proc_NFKC_Casefold
    utf8proc_NFKD=sherpa_onnx_utf8proc_NFKD
    utf8proc_normalize_utf32=sherpa_onnx_utf8proc_normalize_utf32
    utf8proc_reencode=sherpa_onnx_utf8proc_reencode
    utf8proc_tolower=sherpa_onnx_utf8proc_tolower
    utf8proc_totitle=sherpa_onnx_utf8proc_totitle
    utf8proc_toupper=sherpa_onnx_utf8proc_toupper
    utf8proc_unicode_version=sherpa_onnx_utf8proc_unicode_version
    utf8proc_version=sherpa_onnx_utf8proc_version
  )

  # utf8proc renames the static library to utf8proc_static under MSVC, which
  # would make the artifact name platform-dependent. Force it back to
  # "utf8proc" everywhere so -lutf8proc / utf8proc.lib / {$linklib utf8proc}
  # and the Rust static-lib list all stay correct on every OS.
  set_target_properties(utf8proc PROPERTIES OUTPUT_NAME utf8proc)

  if(_build_shared_libs_bak)
    set_target_properties(utf8proc
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS utf8proc DESTINATION lib)
  endif()
endfunction()

download_utf8proc()
