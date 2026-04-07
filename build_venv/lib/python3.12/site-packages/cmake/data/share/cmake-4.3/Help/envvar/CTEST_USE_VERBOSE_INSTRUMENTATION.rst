CTEST_USE_VERBOSE_INSTRUMENTATION
---------------------------------

.. versionadded:: 4.3

.. include:: include/ENV_VAR.rst

Setting this environment variable to ``1``, ``True``, or ``ON`` causes CTest to
report the full command line (including arguments) to CDash for each
instrumented command. By default, CTest truncates the command line at the first
space.

See also :envvar:`CTEST_USE_INSTRUMENTATION`
