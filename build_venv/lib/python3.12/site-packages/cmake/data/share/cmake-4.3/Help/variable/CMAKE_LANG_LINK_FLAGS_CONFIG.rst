CMAKE_<LANG>_LINK_FLAGS_<CONFIG>
--------------------------------

.. versionadded:: 4.3

Language-wide flags for language ``<LANG>`` used when linking for the
``<CONFIG>`` configuration. These flags will be passed to all invocations of
the compiler which drive linking.

See :variable:`CMAKE_<LANG>_LINK_FLAGS` for the ordering of these flags with
respect to other variables. Notably, flags in
``CMAKE_<LANG>_LINK_FLAGS_<CONFIG>`` are passed after those in
:variable:`CMAKE_<LANG>_LINK_FLAGS`.

.. include:: ../command/include/LINK_LIBRARIES_LINKER.rst

This support implies to parse and re-quote the content of the variable.

See Also
^^^^^^^^

* :variable:`CMAKE_<LANG>_LINK_FLAGS`
* :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>`
