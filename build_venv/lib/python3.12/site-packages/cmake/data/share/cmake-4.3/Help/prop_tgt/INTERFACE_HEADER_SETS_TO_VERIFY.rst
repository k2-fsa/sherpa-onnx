INTERFACE_HEADER_SETS_TO_VERIFY
-------------------------------

.. versionadded:: 3.24

Used to specify which ``PUBLIC`` and ``INTERFACE`` header sets of a target
should be verified as interface headers.

This property contains a semicolon-separated list of header sets which
should be verified if :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS` is set to true.
If the list is empty, all ``PUBLIC`` and ``INTERFACE`` header sets are
verified. If the project does not want to verify any interface header sets on
the target, set :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS` to false.

See also :prop_tgt:`PRIVATE_HEADER_SETS_TO_VERIFY`.
