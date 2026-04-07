PRIVATE_HEADER_SETS_TO_VERIFY
-----------------------------

.. versionadded:: 4.3

Used to specify which ``PUBLIC`` and ``PRIVATE`` header sets of a target
should be verified as private headers.

This property contains a semicolon-separated list of header sets which
should be verified if :prop_tgt:`VERIFY_PRIVATE_HEADER_SETS` is set to true.
If the list is empty, all ``PUBLIC`` and ``PRIVATE`` header sets are verified.
If the project does not want to verify any private header sets on the target,
set :prop_tgt:`VERIFY_PRIVATE_HEADER_SETS` to false.

See also :prop_tgt:`INTERFACE_HEADER_SETS_TO_VERIFY`.
