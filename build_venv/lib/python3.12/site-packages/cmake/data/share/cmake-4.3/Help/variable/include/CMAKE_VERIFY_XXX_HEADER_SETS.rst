This variable is used to initialize the |VERIFY_XXX_HEADER_SETS| property of
targets when they are created.  Setting it to true enables |xxx| header set
verification.

Projects should not normally set this variable, it is intended as a developer
control to be set on the :manual:`cmake(1)` command line or other
equivalent methods.  The developer must have the ability to enable or
disable header set verification according to the capabilities of their own
machine and compiler.

Verification of a dependency's header sets is not typically of interest to
developers.  Therefore, :command:`FetchContent_MakeAvailable` explicitly sets
|CMAKE_VERIFY_XXX_HEADER_SETS| and |COMPLEMENTARY_CMAKE_VERIFY_XXX_HEADER_SETS|
to false for the duration of its call, but restores their original values
before returning.  If a project brings a dependency directly into the main
build (e.g. calling :command:`add_subdirectory` on a vendored project from a
git submodule), it should also do likewise.  For example:

.. code:: cmake

  # Save original setting so we can restore it later
  set(want_interface_header_set_verification ${CMAKE_VERIFY_INTERFACE_HEADER_SETS})
  set(want_private_header_set_verification ${CMAKE_VERIFY_PRIVATE_HEADER_SETS})

  # Include the vendored dependency with header set verification disabled
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS OFF)
  set(CMAKE_VERIFY_PRIVATE_HEADER_SETS OFF)
  add_subdirectory(...)   # Vendored sources, e.g. from git submodules

  # Add the project's own sources. Restore the developer's original choice
  # for whether to enable header set verification.
  set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ${want_interface_header_set_verification})
  set(CMAKE_VERIFY_PRIVATE_HEADER_SETS ${want_private_header_set_verification})
  add_subdirectory(src)

By default, this variable is not set, which will result in |xxx| header set
verification being disabled.

See also |COMPLEMENTARY_CMAKE_VERIFY_XXX_HEADER_SETS|.
