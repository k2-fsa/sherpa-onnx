function(download_portaudio)
  include(FetchContent)

  set(portaudio_URL  "http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz")
  set(portaudio_URL2 "https://huggingface.co/csukuangfj/sherpa-onnx-cmake-deps/resolve/main/pa_stable_v190700_20210406.tgz")
  set(portaudio_HASH "SHA256=47efbf42c77c19a05d22e627d42873e991ec0c1357219c0d74ce6a2948cb2def")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  set(possible_file_locations
    $ENV{HOME}/Downloads/pa_stable_v190700_20210406.tgz
    $ENV{HOME}/asr/pa_stable_v190700_20210406.tgz
    ${PROJECT_SOURCE_DIR}/pa_stable_v190700_20210406.tgz
    ${PROJECT_BINARY_DIR}/pa_stable_v190700_20210406.tgz
    /tmp/pa_stable_v190700_20210406.tgz
    /star-fj/fangjun/download/github/pa_stable_v190700_20210406.tgz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(portaudio_URL  "file://${f}")
      break()
    endif()
  endforeach()

  if(BUILD_SHARED_LIBS)
    set(PA_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(PA_BUILD_STATIC OFF CACHE BOOL "" FORCE)
  else()
    set(PA_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(PA_BUILD_STATIC ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_Declare(portaudio
    URL
      ${portaudio_URL}
      ${portaudio_URL2}
    URL_HASH          ${portaudio_HASH}
  )

  FetchContent_GetProperties(portaudio)
  if(NOT portaudio_POPULATED)
    message(STATUS "Downloading portaudio from ${portaudio_URL}")
    FetchContent_Populate(portaudio)
  endif()
  message(STATUS "portaudio is downloaded to ${portaudio_SOURCE_DIR}")
  message(STATUS "portaudio's binary dir is ${portaudio_BINARY_DIR}")

  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON) # to solve the following warning on macOS
  endif()

  add_subdirectory(${portaudio_SOURCE_DIR} ${portaudio_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_portaudio()

# Note
# See http://portaudio.com/docs/v19-doxydocs/tutorial_start.html
# for how to use portaudio
