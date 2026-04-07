This specifies that the current CMake code is written for the given range of
CMake versions, ``<min>[...<policy_max>]``. It sets the "policy version" to:

* the range's ``<policy_max>`` version, if specified, or to
* the ``<min>`` version, or to
* the value of the :variable:`CMAKE_POLICY_VERSION_MINIMUM` variable
  if it is higher than the other two versions.

The policy version effectively requests behavior preferred as of a given CMake
version and tells newer CMake versions to warn about their new policies.
All policies known to the running version of CMake and introduced
in the policy version or earlier will be set to use ``NEW`` behavior.
All policies introduced in later versions will be unset (unless the
:variable:`CMAKE_POLICY_DEFAULT_CMP<NNNN>` variable sets a default).

.. note::

  ``...<policy_max>`` does *not* signify that later CMake versions are
  forbidden.  It merely specifies the highest CMake version for which
  the project or module has been actively updated and maintained.
