# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindRuby
--------

Finds Ruby installation and the locations of its include files and libraries:

.. code-block:: cmake

  find_package(Ruby [<version>] [COMPONENTS <components>...] [...])

Ruby is a general-purpose programming language.  This module supports Ruby
2.0 through 4.0.  Virtual environments, such as RVM or RBENV, are also
supported.

Components
^^^^^^^^^^

.. versionadded:: 4.3

This module supports the following components:

``Interpreter``
  The Ruby interpreter executable.

``Development``
  Headers and libraries needed to build Ruby extensions or embed Ruby.

If no components are specified, both ``Interpreter`` and ``Development``
are searched for.

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 4.3

This module defines the following :prop_tgt:`IMPORTED` targets:

``Ruby::Interpreter``
  Ruby interpreter. Target defined if component ``Interpreter`` is found.

``Ruby::Ruby``
  Ruby library for embedding Ruby in C/C++ applications.
  Target defined if component ``Development`` is found.

``Ruby::Module``
  Ruby library for building Ruby extension modules.
  Target defined if component ``Development`` is found.
  Use this target when creating native extensions that will be
  loaded into Ruby via ``require``. On most platforms, extension
  modules do not link directly to libruby. Includes appropriate
  symbol visibility settings.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Ruby_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) ruby was found.

``Ruby_VERSION``
  The version of ruby which was found, e.g. ``3.2.6``.

``Ruby_VERSION_MAJOR``
  Ruby major version.

``Ruby_VERSION_MINOR``
  Ruby minor version.

``Ruby_VERSION_PATCH``
  Ruby patch version.

``Ruby_EXECUTABLE``
  The full path to the ruby binary.

``Ruby_INCLUDE_DIRS``
  Include dirs to be used when using the ruby library.

``Ruby_LIBRARIES``
  .. versionadded:: 3.18

  Libraries needed to use ruby from C.

.. versionchanged:: 3.18
  Previous versions of CMake used the ``RUBY_`` prefix for all variables.

Hints
^^^^^

This module accepts the following variables:

``Ruby_FIND_VIRTUALENV``
  .. versionadded:: 3.18

  This variable defines the handling of virtual environments.
  It can be left empty or be set to one of the following values:

  * ``FIRST``: Virtual Ruby environments are searched for first,
               then the system Ruby installation.
               This is the default.
  * ``ONLY``: Only virtual environments are searched
  * ``STANDARD``: Only the system Ruby installation is searched.

  Virtual environments may be provided by:

  ``rvm``
    Requires that the ``MY_RUBY_HOME`` environment is defined.

  ``rbenv``
    Requires that ``rbenv`` is installed in ``~/.rbenv/bin``
    or that the ``RBENV_ROOT`` environment variable is defined.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

.. deprecated:: 4.0
  The following variables are deprecated.  See policy :policy:`CMP0185`.

  ``RUBY_FOUND``
    Same as ``Ruby_FOUND``.
  ``RUBY_VERSION``
    Same as ``Ruby_VERSION``.
  ``RUBY_EXECUTABLE``
    Same as ``Ruby_EXECUTABLE``.
  ``RUBY_INCLUDE_DIRS``
    Same as ``Ruby_INCLUDE_DIRS``.
  ``RUBY_INCLUDE_PATH``
    Same as ``Ruby_INCLUDE_DIRS``.
  ``RUBY_LIBRARY``
    Same as ``Ruby_LIBRARY``.

Examples
^^^^^^^^

Finding Ruby and specifying the minimum required version:

.. code-block:: cmake

  find_package(Ruby 3.2.6 EXACT REQUIRED)
  # or
  find_package(Ruby 3.2)
#]=======================================================================]

cmake_policy(GET CMP0185 _Ruby_CMP0185)

if(NOT _Ruby_CMP0185 STREQUAL "NEW")
  # Backwards compatibility
  # Define camel case versions of input variables
  foreach (UPPER
           RUBY_EXECUTABLE
           RUBY_LIBRARY
           RUBY_INCLUDE_DIR
           RUBY_CONFIG_INCLUDE_DIR)
    if (DEFINED ${UPPER})
      string(REPLACE "RUBY_" "Ruby_" Camel ${UPPER})
      if (NOT DEFINED ${Camel})
        set(${Camel} ${${UPPER}})
      endif ()
    endif ()
  endforeach ()
endif()

# Uncomment the following line to get debug output for this file
# set(CMAKE_MESSAGE_LOG_LEVEL DEBUG)

# Determine the list of possible names of the ruby executable depending
# on which version of ruby is required
set(_Ruby_POSSIBLE_EXECUTABLE_NAMES ruby)

# If the user has not specified a Ruby version, create a list of Ruby versions
# to search (newest to oldest). Based on https://www.ruby-lang.org/en/downloads/releases/
if (NOT Ruby_FIND_VERSION_EXACT)
  set(_Ruby_SUPPORTED_VERSIONS 40 34 33 32)
  set(_Ruby_UNSUPPORTED_VERSIONS 31 30 27 26 25 24 23 22 21 20)
  foreach (_ruby_version IN LISTS _Ruby_SUPPORTED_VERSIONS _Ruby_UNSUPPORTED_VERSIONS)
    string(SUBSTRING "${_ruby_version}" 0 1 _ruby_major_version)
    string(SUBSTRING "${_ruby_version}" 1 1 _ruby_minor_version)
    # Append both rubyX.Y and rubyXY (eg: ruby3.4 ruby34)
    list(APPEND _Ruby_POSSIBLE_EXECUTABLE_NAMES
         ruby${_ruby_major_version}.${_ruby_minor_version}
         ruby${_ruby_major_version}${_ruby_minor_version})
  endforeach ()
endif ()

# Virtual environment handling
if (DEFINED Ruby_FIND_VIRTUALENV AND NOT Ruby_FIND_VIRTUALENV MATCHES "^(FIRST|ONLY|STANDARD)$")
  message(AUTHOR_WARNING "FindRuby: ${Ruby_FIND_VIRTUALENV}: invalid value for 'Ruby_FIND_VIRTUALENV'. 'FIRST', 'ONLY' or 'STANDARD' expected. 'FIRST' will be used instead.")
  set(Ruby_FIND_VIRTUALENV "FIRST")
elseif (NOT DEFINED Ruby_FIND_VIRTUALENV)
  # Default is to search for virtual environments first
  set(Ruby_FIND_VIRTUALENV "FIRST")
endif ()

# Validate the found Ruby interpreter to make sure that it is
# callable and that its version matches the requested version
function(_RUBY_VALIDATE_INTERPRETER result_var path)
  # Get the interpreter version
  execute_process(COMMAND "${path}" -e "puts RUBY_VERSION"
                  RESULT_VARIABLE result
                  OUTPUT_VARIABLE version
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (NOT result EQUAL 0)
    set(_Ruby_Interpreter_REASON_FAILURE "Cannot use the interpreter \"${path}\"")
    set(${result_var} FALSE PARENT_SCOPE)
    return()
  endif ()

  if (Ruby_FIND_VERSION)
    if (Ruby_FIND_VERSION_EXACT AND NOT version VERSION_EQUAL Ruby_FIND_VERSION)
      message(DEBUG "Incorrect Ruby found. Requested: ${Ruby_FIND_VERSION}. Found: ${version}. Path: \"${path}\"")
      set(${result_var} FALSE PARENT_SCOPE)
      return()
    elseif (version VERSION_LESS Ruby_FIND_VERSION)
      message(DEBUG "Ruby version is too old. Minimum: ${Ruby_FIND_VERSION}. Found: ${version}. Path: \"${path}\"")
      set(${result_var} FALSE PARENT_SCOPE)
      return()
    endif ()
  endif ()

  # Found valid Ruby interpreter!
  set(${result_var} TRUE PARENT_SCOPE)
endfunction()

# Query Ruby RBConfig module for the specified variable (_RUBY_CONFIG_VAR)
function(_RUBY_CONFIG_VAR RBVAR OUTVAR)
  execute_process(COMMAND "${Ruby_EXECUTABLE}" -r rbconfig -e "print RbConfig::CONFIG['${RBVAR}']"
                  RESULT_VARIABLE _Ruby_SUCCESS
                  OUTPUT_VARIABLE _Ruby_OUTPUT
                  ERROR_QUIET)

  set(${OUTVAR} "${_Ruby_OUTPUT}" PARENT_SCOPE)
endfunction()

# Check for RVM virtual environments
function(_RUBY_CHECK_RVM)
  if (NOT DEFINED ENV{MY_RUBY_HOME})
    return()
  endif ()

  find_program(Ruby_EXECUTABLE
               NAMES ${_Ruby_POSSIBLE_EXECUTABLE_NAMES}
               NAMES_PER_DIR
               PATHS ENV MY_RUBY_HOME
               PATH_SUFFIXES bin Scripts
               VALIDATOR _RUBY_VALIDATE_INTERPRETER
               NO_CMAKE_PATH
               NO_CMAKE_ENVIRONMENT_PATH
               NO_SYSTEM_ENVIRONMENT_PATH
               NO_CMAKE_SYSTEM_PATH)

  if (Ruby_EXECUTABLE)
    set(Ruby_ENV "RVM" CACHE INTERNAL "Ruby environment")
  endif ()
endfunction()

# Check for RBENV virtual environments
function(_RUBY_CHECK_RBENV)
  find_program(Ruby_RBENV_EXECUTABLE
               NAMES rbenv
               NAMES_PER_DIR
               PATHS "$ENV{HOME}/.rbenv/bin/rbenv" ENV RBENV_ROOT
               PATH_SUFFIXES bin Scripts
               NO_CACHE
               NO_CMAKE_PATH
               NO_CMAKE_ENVIRONMENT_PATH
               NO_CMAKE_SYSTEM_PATH)

  execute_process(COMMAND "${Ruby_RBENV_EXECUTABLE}" "which" "ruby"
                  RESULT_VARIABLE result
                  OUTPUT_VARIABLE ruby_exe
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT result EQUAL 0)
    return()
  endif ()
  cmake_path(GET ruby_exe PARENT_PATH ruby_dir)

  find_program(Ruby_EXECUTABLE
               NAMES ruby
               NAMES_PER_DIR
               PATHS ${ruby_dir}
               VALIDATOR _RUBY_VALIDATE_INTERPRETER
               NO_DEFAULT_PATH)

  if (Ruby_EXECUTABLE)
    set(Ruby_ENV "RBENV" CACHE INTERNAL "Ruby environment")
  endif ()
endfunction()

# Check Ruby installed via Homebrew on macOS
function(_RUBY_CHECK_BREW)
  # Try to locate brew in common locations and in PATH
  find_program(_BREW_EXECUTABLE
          NAMES brew
          NAMES_PER_DIR
          PATHS
          /opt/homebrew/bin      # Apple Silicon default
          /usr/local/bin         # Intel default
          ENV PATH
          NO_CACHE
  )

  if (NOT _BREW_EXECUTABLE)
    return()
  endif ()

  message(DEBUG "Found brew at: ${_BREW_EXECUTABLE}")

  # Query Homebrew for the prefix of the 'ruby' formula.
  # If Ruby is not installed via Homebrew, this will fail.
  execute_process(
          COMMAND "${_BREW_EXECUTABLE}" --prefix ruby
          RESULT_VARIABLE _Ruby_BREW_RESULT
          OUTPUT_VARIABLE _Ruby_BREW_DIR
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(DEBUG "Ruby BREW is: ${_Ruby_BREW_DIR}")

  if (NOT _Ruby_BREW_RESULT EQUAL 0 OR _Ruby_BREW_DIR STREQUAL "")
    # No 'ruby' formula installed in Homebrew
    return()
  endif ()

  find_program(Ruby_EXECUTABLE
          NAMES ${_Ruby_POSSIBLE_EXECUTABLE_NAMES}
          NAMES_PER_DIR
          PATHS "${_Ruby_BREW_DIR}/bin"
          VALIDATOR _RUBY_VALIDATE_INTERPRETER
          NO_DEFAULT_PATH
  )

  if (Ruby_EXECUTABLE)
    set(Ruby_ENV "BREW" CACHE INTERNAL "Ruby environment")
  endif ()
endfunction()

# Check system installed Ruby
function(_RUBY_CHECK_SYSTEM)
  find_program(Ruby_EXECUTABLE
               NAMES ${_Ruby_POSSIBLE_EXECUTABLE_NAMES}
               NAMES_PER_DIR
               VALIDATOR _RUBY_VALIDATE_INTERPRETER)

  if (Ruby_EXECUTABLE)
    set(Ruby_ENV "Standard" CACHE INTERNAL "Ruby environment")
  endif ()
endfunction()

# Find Ruby
if (NOT Ruby_EXECUTABLE AND Ruby_FIND_VIRTUALENV MATCHES "^(FIRST|ONLY)$")
  # First check for RVM virtual environments
  _RUBY_CHECK_RVM()
  # Second check for RBENV virtual environments
  if (NOT Ruby_EXECUTABLE)
    _RUBY_CHECK_RBENV()
  endif ()
endif ()

# Check for Homebrew Ruby (non-virtualenv, common on MacOS)
if (NOT Ruby_EXECUTABLE AND NOT Ruby_FIND_VIRTUALENV STREQUAL "ONLY")
  _RUBY_CHECK_BREW()
endif ()

# Fallback to system installed Ruby
if (NOT Ruby_EXECUTABLE AND NOT Ruby_FIND_VIRTUALENV STREQUAL "ONLY")
  _RUBY_CHECK_SYSTEM()
endif ()

# We found a new Ruby or a Ruby that is different than the last one we found.
# So reload a number of variables by querying the Ruby interpreter.
if (Ruby_EXECUTABLE AND NOT Ruby_EXECUTABLE STREQUAL "${_Ruby_EXECUTABLE_LAST_QUERIED}")
  # query the ruby version
  _RUBY_CONFIG_VAR("MAJOR" Ruby_VERSION_MAJOR)
  _RUBY_CONFIG_VAR("MINOR" Ruby_VERSION_MINOR)
  _RUBY_CONFIG_VAR("TEENY" Ruby_VERSION_PATCH)

  # Ruby extensions information
  _RUBY_CONFIG_VAR("arch" Ruby_ARCH) # x86_64-linux, arm64-darwin, x64-mswin64_140, etc
  # Extension directory where so/bundle files are stored
  _RUBY_CONFIG_VAR("archdir" Ruby_ARCH_DIR)
  # Extension suffix
  _RUBY_CONFIG_VAR("DLEXT" _Ruby_DLEXT) # so, bundle, *not* dll

  # Headers
  _RUBY_CONFIG_VAR("rubyhdrdir" Ruby_HDR_DIR)
  _RUBY_CONFIG_VAR("rubyarchhdrdir" Ruby_ARCHHDR_DIR)

  # Ruby library information
  _RUBY_CONFIG_VAR("libdir" _Ruby_POSSIBLE_LIB_DIR) # /usr/lib64
  _RUBY_CONFIG_VAR("RUBY_SO_NAME" _Ruby_SO_NAME) # ruby, x64-vcruntime140-ruby340, etc.

  # Ruby directory for ruby files (*.rb). TODO - not relevant should be removed
  _RUBY_CONFIG_VAR("rubylibdir" Ruby_RUBY_LIB_DIR)

  # site_ruby - TODO - not relevant and should be removed
  _RUBY_CONFIG_VAR("sitearchdir" Ruby_SITEARCH_DIR)
  _RUBY_CONFIG_VAR("sitelibdir" Ruby_SITELIB_DIR)

  # vendor_ruby - TODO - Not relevant and should be removed.
  execute_process(COMMAND "${Ruby_EXECUTABLE}" -r vendor-specific -e "print 'true'"
                  OUTPUT_VARIABLE Ruby_HAS_VENDOR_RUBY ERROR_QUIET)

  if (Ruby_HAS_VENDOR_RUBY)
    _RUBY_CONFIG_VAR("vendorlibdir" Ruby_VENDORLIB_DIR)
    _RUBY_CONFIG_VAR("vendorarchdir" Ruby_VENDORARCH_DIR)
  endif ()

  # save the results in the cache so we don't have to run ruby the next time again
  set(_Ruby_EXECUTABLE_LAST_QUERIED "${Ruby_EXECUTABLE}" CACHE INTERNAL "The ruby executable last queried for version and path info")
  set(Ruby_VERSION_MAJOR ${Ruby_VERSION_MAJOR} CACHE STRING "The Ruby major version" FORCE)
  set(Ruby_VERSION_MINOR ${Ruby_VERSION_MINOR} CACHE STRING "The Ruby minor version" FORCE)
  set(Ruby_VERSION_PATCH ${Ruby_VERSION_PATCH} CACHE STRING "The Ruby patch version" FORCE)
  set(Ruby_ARCH_DIR ${Ruby_ARCH_DIR} CACHE INTERNAL "The Ruby arch dir" FORCE)
  set(Ruby_HDR_DIR ${Ruby_HDR_DIR} CACHE INTERNAL "The Ruby header dir (1.9+)" FORCE)
  set(Ruby_ARCHHDR_DIR ${Ruby_ARCHHDR_DIR} CACHE INTERNAL "The Ruby arch header dir (2.0+)" FORCE)
  set(_Ruby_POSSIBLE_LIB_DIR ${_Ruby_POSSIBLE_LIB_DIR} CACHE INTERNAL "The Ruby lib dir" FORCE)
  set(Ruby_RUBY_LIB_DIR ${Ruby_RUBY_LIB_DIR} CACHE INTERNAL "The Ruby ruby-lib dir" FORCE)
  set(_Ruby_SO_NAME ${_Ruby_SO_NAME} CACHE STRING "The Ruby shared library name" FORCE)
  set(_Ruby_DLEXT ${_Ruby_DLEXT} CACHE STRING "Ruby extensions extension" FORCE)
  set(Ruby_SITEARCH_DIR ${Ruby_SITEARCH_DIR} CACHE INTERNAL "The Ruby site arch dir" FORCE)
  set(Ruby_SITELIB_DIR ${Ruby_SITELIB_DIR} CACHE INTERNAL "The Ruby site lib dir" FORCE)
  set(Ruby_HAS_VENDOR_RUBY ${Ruby_HAS_VENDOR_RUBY} CACHE BOOL "Vendor Ruby is available" FORCE)
  set(Ruby_VENDORARCH_DIR ${Ruby_VENDORARCH_DIR} CACHE INTERNAL "The Ruby vendor arch dir" FORCE)
  set(Ruby_VENDORLIB_DIR ${Ruby_VENDORLIB_DIR} CACHE INTERNAL "The Ruby vendor lib dir" FORCE)

  mark_as_advanced(
      Ruby_ARCH_DIR
      Ruby_ARCH
      Ruby_HDR_DIR
      Ruby_ARCHHDR_DIR
      _Ruby_POSSIBLE_LIB_DIR
      Ruby_RUBY_LIB_DIR
      _Ruby_SO_NAME
      _Ruby_DLEXT
      Ruby_SITEARCH_DIR
      Ruby_SITELIB_DIR
      Ruby_HAS_VENDOR_RUBY
      Ruby_VENDORARCH_DIR
      Ruby_VENDORLIB_DIR
      Ruby_VERSION_MAJOR
      Ruby_VERSION_MINOR
      Ruby_VERSION_PATCH
  )
endif ()

if (Ruby_VERSION_MAJOR)
  set(Ruby_VERSION "${Ruby_VERSION_MAJOR}.${Ruby_VERSION_MINOR}.${Ruby_VERSION_PATCH}")
  set(_Ruby_VERSION_NODOT "${Ruby_VERSION_MAJOR}${Ruby_VERSION_MINOR}${Ruby_VERSION_PATCH}")
  set(_Ruby_VERSION_NODOT_ZERO_PATCH "${Ruby_VERSION_MAJOR}${Ruby_VERSION_MINOR}0")
  set(_Ruby_VERSION_SHORT "${Ruby_VERSION_MAJOR}.${Ruby_VERSION_MINOR}")
  set(_Ruby_VERSION_SHORT_NODOT "${Ruby_VERSION_MAJOR}${Ruby_VERSION_MINOR}")
endif ()

# Save CMAKE_FIND_FRAMEWORK
set(_Ruby_CMAKE_FIND_FRAMEWORK_ORIGINAL ${CMAKE_FIND_FRAMEWORK})

# Avoid finding the ancient Ruby framework included in macOS.
set(CMAKE_FIND_FRAMEWORK LAST)

find_path(Ruby_INCLUDE_DIR
        NAMES ruby.h
        HINTS ${Ruby_HDR_DIR})

find_path(Ruby_CONFIG_INCLUDE_DIR
        NAMES ruby/config.h config.h
        HINTS ${Ruby_ARCHHDR_DIR})

set(Ruby_INCLUDE_DIRS ${Ruby_INCLUDE_DIR} ${Ruby_CONFIG_INCLUDE_DIR})

find_library(Ruby_LIBRARY
        NAMES "${_Ruby_SO_NAME}"
        HINTS ${_Ruby_POSSIBLE_LIB_DIR})

# Restore CMAKE_FIND_FRAMEWORK
set(CMAKE_FIND_FRAMEWORK ${_Ruby_CMAKE_FIND_FRAMEWORK_ORIGINAL})

message(DEBUG "--------FindRuby.cmake debug------------")
message(DEBUG "_Ruby_POSSIBLE_EXECUTABLE_NAMES: ${_Ruby_POSSIBLE_EXECUTABLE_NAMES}")
message(DEBUG "_Ruby_POSSIBLE_LIB_DIR: ${_Ruby_POSSIBLE_LIB_DIR}")
message(DEBUG "_Ruby_SO_NAME: ${_Ruby_SO_NAME}")
message(DEBUG "_Ruby_DLEXT: ${_Ruby_DLEXT}")
message(DEBUG "Ruby_FIND_VIRTUALENV=${Ruby_FIND_VIRTUALENV}")
message(DEBUG "Ruby_ENV: ${Ruby_ENV}")
message(DEBUG "Found Ruby_VERSION: \"${Ruby_VERSION}\"")
message(DEBUG "Ruby_EXECUTABLE: ${Ruby_EXECUTABLE}")
message(DEBUG "Ruby_LIBRARY: ${Ruby_LIBRARY}")
message(DEBUG "Ruby_INCLUDE_DIR: ${Ruby_INCLUDE_DIR}")
message(DEBUG "Ruby_CONFIG_INCLUDE_DIR: ${Ruby_CONFIG_INCLUDE_DIR}")
message(DEBUG "Ruby_HDR_DIR: ${Ruby_HDR_DIR}")
message(DEBUG "Ruby_ARCH_DIR: ${Ruby_ARCH_DIR}")
message(DEBUG "Ruby_ARCHHDR_DIR: ${Ruby_ARCHHDR_DIR}")
message(DEBUG "--------------------")

# Components
#
# If the caller does not request components, preserve legacy behavior
if (NOT Ruby_FIND_COMPONENTS)
  set(Ruby_FIND_COMPONENTS Interpreter Development)
endif ()

set(_Ruby_WANT_INTERPRETER FALSE)
set(_Ruby_WANT_DEVELOPMENT FALSE)
set(_Ruby_REQUIRED_VARS "")

foreach (component IN LISTS Ruby_FIND_COMPONENTS)
  if (component STREQUAL "Interpreter")
    set(_Ruby_WANT_INTERPRETER TRUE)
    list(APPEND _Ruby_REQUIRED_VARS Ruby_EXECUTABLE)
  elseif (component STREQUAL "Development")
    set(_Ruby_WANT_DEVELOPMENT TRUE)
    list(APPEND _Ruby_REQUIRED_VARS Ruby_INCLUDE_DIR Ruby_CONFIG_INCLUDE_DIR)
    if (WIN32)
      list(APPEND _Ruby_REQUIRED_VARS Ruby_LIBRARY)
    endif ()
  else ()
    message(FATAL_ERROR
            "FindRuby: Unsupported component '${component}'. Supported components are: Interpreter, Development")
  endif ()
endforeach ()

# Set component found flags
if (Ruby_EXECUTABLE)
  set(Ruby_Interpreter_FOUND TRUE)
else ()
  set(Ruby_Interpreter_FOUND FALSE)
endif ()

if (Ruby_INCLUDE_DIR AND Ruby_CONFIG_INCLUDE_DIR AND (Ruby_LIBRARY OR NOT WIN32))
  set(Ruby_Development_FOUND TRUE)
else ()
  set(Ruby_Development_FOUND FALSE)
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ruby
                                  REQUIRED_VARS ${_Ruby_REQUIRED_VARS}
                                  VERSION_VAR Ruby_VERSION
                                  HANDLE_COMPONENTS)

if (Ruby_FOUND)
  if (NOT TARGET Ruby::Interpreter)
    add_executable(Ruby::Interpreter IMPORTED GLOBAL)
    set_target_properties(Ruby::Interpreter PROPERTIES
      IMPORTED_LOCATION "${Ruby_EXECUTABLE}"
    )
  endif ()

  if (Ruby_Development_FOUND)
    set(Ruby_LIBRARIES ${Ruby_LIBRARY})

    if (Ruby_LIBRARY AND NOT TARGET Ruby::Ruby)
      add_library(Ruby::Ruby UNKNOWN IMPORTED)
      set_target_properties(Ruby::Ruby PROPERTIES
        IMPORTED_LOCATION "${Ruby_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Ruby_INCLUDE_DIRS}"
        # Custom property for extension suffix (with dot), e.g. ".so", ".bundle"
        INTERFACE_RUBY_EXTENSION_SUFFIX ".${_Ruby_DLEXT}"
      )
    endif ()

    # Ruby::Module - For building Ruby extension modules
    if (NOT TARGET Ruby::Module)
      if (WIN32)
        add_library(Ruby::Module UNKNOWN IMPORTED)
        set_target_properties(Ruby::Module PROPERTIES
          IMPORTED_LOCATION "${Ruby_LIBRARY}"
        )
      else ()
        add_library(Ruby::Module INTERFACE IMPORTED)
      endif ()

      set_target_properties(Ruby::Module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Ruby_INCLUDE_DIRS}"
        # Custom property for extension suffix (with dot), e.g. ".so", ".bundle"
        INTERFACE_RUBY_EXTENSION_SUFFIX ".${_Ruby_DLEXT}"
        INTERFACE_C_VISIBILITY_PRESET hidden
        INTERFACE_CXX_VISIBILITY_PRESET hidden
        INTERFACE_VISIBILITY_INLINES_HIDDEN ON
      )

      # macOS: allow unresolved Ruby API symbols; resolved when Ruby loads the bundle.
      if (APPLE)
        target_link_options(Ruby::Module INTERFACE
          "LINKER:-undefined,dynamic_lookup"
        )
      endif ()

      # Linux (and other ELF platforms):
      # Normally undefined Ruby API symbols are allowed in shared objects and resolved at dlopen().
      # But if the toolchain/preset adds -Wl,--no-undefined, linking will fail.
      # This counteracts that.
      if (UNIX AND NOT APPLE)
        target_link_options(Ruby::Module INTERFACE
          "LINKER:--unresolved-symbols=ignore-all"
        )
      endif ()
    endif ()
  endif ()
endif ()

mark_as_advanced(
    Ruby_EXECUTABLE
    Ruby_LIBRARY
    Ruby_INCLUDE_DIR
    Ruby_CONFIG_INCLUDE_DIR
)

if(NOT _Ruby_CMP0185 STREQUAL "NEW")
  # Set some variables for compatibility with previous version of this file (no need to provide a CamelCase version of that...)
  set(RUBY_POSSIBLE_LIB_PATH ${_Ruby_POSSIBLE_LIB_DIR})
  set(RUBY_RUBY_LIB_PATH ${Ruby_RUBY_LIB_DIR})
  set(RUBY_INCLUDE_PATH ${Ruby_INCLUDE_DIRS})

  # Backwards compatibility
  # Define upper case versions of output variables
  foreach (Camel
           Ruby_EXECUTABLE
           Ruby_INCLUDE_DIRS
           Ruby_LIBRARY
           Ruby_VERSION
           Ruby_VERSION_MAJOR
           Ruby_VERSION_MINOR
           Ruby_VERSION_PATCH

           Ruby_ARCH_DIR
           Ruby_ARCH
           Ruby_HDR_DIR
           Ruby_ARCHHDR_DIR
           Ruby_RUBY_LIB_DIR
           Ruby_SITEARCH_DIR
           Ruby_SITELIB_DIR
           Ruby_HAS_VENDOR_RUBY
           Ruby_VENDORARCH_DIR
           Ruby_VENDORLIB_DIR)
    string(TOUPPER ${Camel} UPPER)
    set(${UPPER} ${${Camel}})
  endforeach ()
endif()
