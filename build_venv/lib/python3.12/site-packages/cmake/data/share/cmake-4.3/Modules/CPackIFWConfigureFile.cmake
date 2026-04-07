# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CPackIFWConfigureFile
---------------------

.. versionadded:: 3.8

This module provides a command similar to :command:`configure_file` for
configuring file templates prepared in QtIFW/SDK/Creator style.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CPackIFWConfigureFile)

Commands
^^^^^^^^

This module provides the following command:

.. command:: cpack_ifw_configure_file

  Copies a file template to output file and substitutes variable values
  referenced as ``%{VAR}`` or ``%VAR%`` from the input file template
  content:

  .. code-block:: cmake

    cpack_ifw_configure_file(<input> <output>)

  ``<input>``
    Input file template.  If given as a relative path, it is interpreted as
    relative to the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).

  ``<output>``
    Output file.  If given as a relative path, it is interpreted as relative
    to the current binary directory (:variable:`CMAKE_CURRENT_BINARY_DIR`).

  Qt Installer Framework (QtIFW) uses ``@`` characters for embedding
  predefined variables (``TargetDir``, ``StartMenuDir``, etc.) in Qt
  installer scripts:

  .. code-block:: javascript
    :caption: ``example.qs``

    component.addOperation(
      "CreateShortcut",
      "@TargetDir@/example.com.html",
      "@StartMenuDir@/Example Web Site.lnk"
    );

  The purpose of this command is to preserve the QtIFW predefined variables
  containing the ``@`` characters (``@VAR@``), and instead use the ``%``
  characters for template placeholders (``%VAR%``, ``%{VAR}``) in
  Qt/IFW/SDK/Creator templates.  The :command:`configure_file` command
  would otherwise replace all variable references containing the ``@``
  characters.

  Each variable reference will be replaced with the current value of the
  variable, or the empty string if the variable is not defined.

Examples
^^^^^^^^

In the following example this module is used to create an IFW component
script from a given template file ``qt.tools.foo.qs.in``, where
``%FOO_DOC_DIR%`` variable reference will be replaced by the values of
the ``FOO_DOC_DIR`` CMake variable.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  cmake_minimum_required(VERSION 3.8)

  project(Foo)

  # ...

  include(CPackIFWConfigureFile)

  set(FOO_DOC_DIR "doc/foo")

  cpack_ifw_configure_file(qt.tools.foo.qs.in qt.tools.foo.qs)

.. code-block:: javascript
  :caption: ``qt.tools.foo.qs.in``

  function Component()
  {
  }

  Component.prototype.createOperations = function()
  {
    if (installer.value("os") === "win") {
      component.addOperation(
        "CreateShortcut",
        "@TargetDir@/%FOO_DOC_DIR%/example.com.html",
        "@StartMenuDir@/Example Web Site.lnk"
      );
    }

    component.createOperations();
  }

  // ...

See Also
^^^^^^^^

* The :cpack_gen:`CPack IFW Generator`.
* The :module:`CPackIFW` module.
#]=======================================================================]

if(NOT DEFINED CPackIFWConfigureFile_CMake_INCLUDED)
set(CPackIFWConfigureFile_CMake_INCLUDED 1)

macro(cpack_ifw_configure_file INPUT OUTPUT)
  file(READ "${INPUT}" _tmp)
  foreach(_tmp_regex "%{([^%}]+)}" "%([^%]+)%")
    string(REGEX MATCHALL "${_tmp_regex}" _tmp_vars "${_tmp}")
    while(_tmp_vars)
      foreach(_tmp_var ${_tmp_vars})
        string(REGEX REPLACE "${_tmp_regex}" "\\1"
          _tmp_var_name "${_tmp_var}")
        if(DEFINED ${_tmp_var_name})
          set(_tmp_var_value "${${_tmp_var_name}}")
        elseif(NOT "$ENV{${_tmp_var_name}}" STREQUAL "")
          set(_tmp_var_value "$ENV{${_tmp_var_name}}")
        else()
          set(_tmp_var_value "")
        endif()
        string(REPLACE "${_tmp_var}" "${_tmp_var_value}" _tmp "${_tmp}")
      endforeach()
      string(REGEX MATCHALL "${_tmp_regex}" _tmp_vars "${_tmp}")
    endwhile()
  endforeach()
  if(IS_ABSOLUTE "${OUTPUT}")
    file(WRITE "${OUTPUT}" "${_tmp}")
  else()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}" "${_tmp}")
  endif()
endmacro()

endif() # NOT DEFINED CPackIFWConfigureFile_CMake_INCLUDED
