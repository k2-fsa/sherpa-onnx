The verification target has its
:prop_tgt:`EXCLUDE_FROM_ALL` and :prop_tgt:`DISABLE_PRECOMPILE_HEADERS`
properties set to true, and its :prop_tgt:`AUTOMOC`, :prop_tgt:`AUTORCC`,
:prop_tgt:`AUTOUIC`, :prop_tgt:`UNITY_BUILD`, and
:prop_tgt:`CXX_SCAN_FOR_MODULES` properties set to false.

If the header's :prop_sf:`LANGUAGE` property is set, the value of that property
is used to determine the language with which to compile the header file.
Otherwise, if the target has any C++ sources, the header is compiled as C++.
Otherwise, if the target has any C sources, the header is compiled as C.
Otherwise, if C++ is enabled globally, the header is compiled as C++.
Otherwise, if C is enabled globally, the header is compiled as C. Otherwise,
the header file is not compiled.

If the header's :prop_sf:`SKIP_LINTING` property is set to true, the file is
not compiled.

If |THIS_PROPERTY| and |COMPLEMENTARY_PROPERTY| are both set to true, headers
belonging to ``PUBLIC`` file sets will be verified twice, but with different
conditions.  The compiler flags used for private and interface contexts can be
different, leading to the compiler interpreting the contents of the header
differently.

If any |xxx| file set verification targets are created, a top-level target
called |THIS_ALL_TARGET| is created which depends on all |xxx| verification
targets.  Another target called ``all_verify_header_sets`` is also created
which depends on |THIS_ALL_TARGET|, and on |COMPLEMENTARY_ALL_TARGET| if it
exists (see |COMPLEMENTARY_PROPERTY|).

This property is initialized by the value of the |INIT_VARIABLE| variable if
it is set when a target is created.

If the project wishes to control which header sets are verified by this
property, it can set |SETS_TO_VERIFY_PROPERTY|.
