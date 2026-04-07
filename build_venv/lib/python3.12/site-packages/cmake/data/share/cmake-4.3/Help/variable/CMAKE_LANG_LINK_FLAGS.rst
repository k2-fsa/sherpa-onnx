CMAKE_<LANG>_LINK_FLAGS
-----------------------

.. versionadded:: 4.3

Language-wide flags for language ``<LANG>`` used when linking for all
configurations. These flags will be passed to all invocations of the compiler
which drive linking.

The flags in this variable will obey the following behavior with respect to
ordering of flags from other variables.

* They will be passed after those added by :variable:`CMAKE_<LANG>_FLAGS` and
  :variable:`CMAKE_<LANG>_FLAGS_<CONFIG>`.
* They will be passed after those added by :variable:`CMAKE_EXE_LINKER_FLAGS`,
  :variable:`CMAKE_EXE_LINKER_FLAGS_<CONFIG>`,
  :variable:`CMAKE_SHARED_LINKER_FLAGS`,
  :variable:`CMAKE_SHARED_LINKER_FLAGS_<CONFIG>`,
  :variable:`CMAKE_MODULE_LINKER_FLAGS`,
  and :variable:`CMAKE_MODULE_LINKER_FLAGS_<CONFIG>` depending on the given
  target type.
* They will be passed before those added by
  :variable:`CMAKE_<LANG>_LINK_FLAGS_<CONFIG>`.
* They will be passed before those added by commands such
  as :command:`add_link_options` and :command:`target_link_options`.

Use of this variable is enabled when policy :policy:`CMP0210` is ``NEW``.

.. include:: ../command/include/LINK_LIBRARIES_LINKER.rst

This support implies to parse and re-quote the content of the variable.

See Also
^^^^^^^^

* :variable:`CMAKE_<LANG>_LINK_FLAGS_<CONFIG>`
* :variable:`CMAKE_<LANG>_FLAGS`
* :policy:`CMP0210`
