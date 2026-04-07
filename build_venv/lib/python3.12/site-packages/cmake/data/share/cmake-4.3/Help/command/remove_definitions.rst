remove_definitions
------------------

Removes compile definitions added by :command:`add_compile_definitions`, or
:command:`add_definitions`:

.. code-block:: cmake

  remove_definitions([<definitions>...])

The arguments are:

``<definitions>...``
  Zero or more compile definitions.

This command can be also used to remove any flags added by
:command:`add_definitions`, but it is intended to remove preprocessor
definitions passed with ``-D``, or ``/D``.

Examples
^^^^^^^^

In the following example targets of the current directory scope will have
only ``BAZ`` and ``QUUX`` compile definitions:

.. code-block:: cmake

  add_compile_definitions(FOO BAR BAZ -DQUUX)

  # ...

  remove_definitions(-DFOO -DBAR)
