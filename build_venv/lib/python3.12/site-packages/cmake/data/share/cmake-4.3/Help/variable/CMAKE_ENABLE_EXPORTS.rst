CMAKE_ENABLE_EXPORTS
--------------------

.. versionadded:: 3.4

.. deprecated:: 4.3
  This variable has been deprecated in favor of the
  :variable:`CMAKE_EXECUTABLE_ENABLE_EXPORTS` and
  :variable:`CMAKE_SHARED_LIBRARY_ENABLE_EXPORTS` variables,
  which have been available since CMake 3.27.
  It is provided for backward compatibility with older CMake code,
  but should not be used in new projects.

Specify whether executables export symbols for loadable modules.

This variable is used to initialize the :prop_tgt:`ENABLE_EXPORTS` target
property for executable targets when they are created by calls to the
:command:`add_executable` command.  See the property documentation for details.
