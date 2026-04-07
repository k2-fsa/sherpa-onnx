IMPORTED_CXX_MODULES_<CONFIG>
-----------------------------

.. versionadded:: 3.28

A list of C++ module specifications available with the target for
configuration ``<CONFIG>``.  Each item in the list is of the form
``<NAME>=<INTERFACE>[,<BMI>]*`` where ``<NAME>`` is the name of the module,
``<INTERFACE>`` is its module interface unit, and any number of available
``<BMI>`` files are provided.

.. note::

   CMake currently does not use the ``<BMI>`` files as there is limited
   support for determining whether an existing ``<BMI>`` file is suitable for
   a given importer of the module it represents.
