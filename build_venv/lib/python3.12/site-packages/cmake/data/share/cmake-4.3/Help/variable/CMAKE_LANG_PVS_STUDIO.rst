CMAKE_<LANG>_PVS_STUDIO
-----------------------

.. versionadded:: 4.3

Default value for :prop_tgt:`<LANG>_PVS_STUDIO` target property
when ``<LANG>`` is ``C`` or ``CXX``.

This variable is used to initialize the property on each target as it is
created.  For example:

.. code-block:: cmake

  set(CMAKE_CXX_PVS_STUDIO pvs-studio-analyzer analyze -a "GA\;OP")
  add_executable(foo foo.cxx)
