# This script copies files with confirmation for overwriting
# It is specifically used for the WASM combined build process in wasm/combined
# and should be kept in the wasm/combined directory.

# Expected variables:
# SRC_DIR - source directory
# DEST_DIR - destination directory
# COPY_FILES - semicolon-separated list of files to copy

# Print debug information
message(STATUS "Source directory: ${SRC_DIR}")
message(STATUS "Destination directory: ${DEST_DIR}")
message(STATUS "Files to copy: ${COPY_FILES}")

# Verify source directory exists
if(NOT EXISTS "${SRC_DIR}")
  message(FATAL_ERROR "Source directory does not exist: ${SRC_DIR}")
endif()

# Verify destination directory exists or create it
if(NOT EXISTS "${DEST_DIR}")
  message(STATUS "Creating destination directory: ${DEST_DIR}")
  file(MAKE_DIRECTORY "${DEST_DIR}")
endif()

# List source directory contents for debugging
message(STATUS "Contents of source directory:")
file(GLOB source_files "${SRC_DIR}/*")
foreach(file ${source_files})
  message(STATUS "  ${file}")
endforeach()

# Process each file (just one file in each call now)
foreach(file_name ${COPY_FILES})
  # Remove quotes if present
  string(REGEX REPLACE "^\"(.*)\"$" "\\1" file_name "${file_name}")
  
  set(src_file "${SRC_DIR}/${file_name}")
  set(dest_file "${DEST_DIR}/${file_name}")
  
  message(STATUS "Processing file: ${file_name}")
  message(STATUS "Source file path: ${src_file}")
  message(STATUS "Destination file path: ${dest_file}")
  
  # Verify source file exists
  if(NOT EXISTS "${src_file}")
    message(FATAL_ERROR "Source file does not exist: ${src_file}")
  endif()
  
  # Check if the destination file exists
  if(EXISTS "${dest_file}")
    message(STATUS "File ${file_name} already exists in ${DEST_DIR}")
    # Prompt for confirmation (this will be shown in terminal)
    message(STATUS "Do you want to overwrite? [y/N]")
    
    # Read user input (works in interactive mode)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E echo_append ""
      COMMAND /bin/bash -c "read -n 1 answer && echo $answer"
      OUTPUT_VARIABLE answer
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if("${answer}" STREQUAL "y" OR "${answer}" STREQUAL "Y")
      message(STATUS "Overwriting ${dest_file}")
      file(COPY "${src_file}" DESTINATION "${DEST_DIR}")
    else()
      message(STATUS "Skipping ${file_name}")
    endif()
  else()
    message(STATUS "Copying ${file_name} to ${DEST_DIR}")
    file(COPY "${src_file}" DESTINATION "${DEST_DIR}")
  endif()
endforeach() 