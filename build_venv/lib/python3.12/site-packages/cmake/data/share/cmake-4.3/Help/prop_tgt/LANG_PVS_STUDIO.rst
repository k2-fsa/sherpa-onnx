<LANG>_PVS_STUDIO
-----------------

.. versionadded:: 4.3

This property is implemented only when ``<LANG>`` is ``C`` or ``CXX``.

Specify a :ref:`semicolon-separated list <CMake Language Lists>` containing
a command line for the ``pvs-studio-analyzer`` tool (named
``CompilerCommandsAnalyzer`` on Windows).  The :ref:`Makefile Generators` and
:ref:`Ninja Generators` will run this tool along with the compiler and
report a warning if the tool reports any problems.

The specified ``pvs-studio-analyzer`` command line will be invoked with
the following additional arguments:

- ``--source-file``: The source file.
- ``--output-file``: A path adjacent to the object file to write the PVS log.
- ``--cl-params``: The compile options.
- ``--preprocessor``: The preprocessor, based on
  :variable:`CMAKE_<LANG>_COMPILER_ID`, if determined to be one of:
  ``visualcpp``, ``clang``, ``gcc``, ``bcc``, ``iar``.
- ``--platform``: The target platform, if determined to be one of: ``arm``,
  ``win32``, ``x64``, ``linux32``, ``linux64``, ``macOS``.

See the
`PVS-Studio documentation <https://pvs-studio.com/en/docs/manual/6615/#flags>`_
for details on these and other available options.

CMake will look for the ``plog-converter`` tool in the same directory as the
provided ``pvs-studio-analyzer``, and in the user's path if not present in that
directory. The ``plog-converter`` will run automatically with the ``Txt``
output type on Windows, and ``errorfile`` on other platforms, and the contents
of that file will be sent to ``stderr``. The PVS log file will be deleted after
the converter runs.

This property is initialized by the value of
the :variable:`CMAKE_<LANG>_PVS_STUDIO` variable if it is set
when a target is created.
