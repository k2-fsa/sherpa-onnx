CMAKE_INSTALL_EXPORTS_AS_PACKAGE_INFO
-------------------------------------

.. versionadded:: 4.3

.. note::

   This variable is meaningful only when experimental support has been enabled
   by the ``CMAKE_EXPERIMENTAL_FIXME`` gate.

A list of directives instructing CMake to install |CPS| package information
when exported target information is installed via :command:`install(EXPORT)`.
The value is treated as a list, with each directive having the form::

  <export-name>:<package-name>[/[l][a<appendix-name>][/<destination>]]

Slashes are used to separate different components of the directive.

Note that this feature is intended for package distributors, and should
**only** be used when editing a project's CMake script is not feasible.
Developers should use :command:`install(PACKAGE_INFO)` directly.

Additionally, because ``CMAKE_INSTALL_EXPORTS_AS_PACKAGE_INFO`` functions by
emulating a call to :command:`install(PACKAGE_INFO)`, using it with a project
that is already calling :command:`install(PACKAGE_INFO)` directly may result
in conflicting installation directives, which will usually cause the project's
configure step to fail.

The meaning of the values is as follows:

``<export-name>``
  Name of the export for which package information should be installed.

``<package-name>``
  Name of the package for which to generate package information.  This is also
  the name that users would use in a :command:`find_package` call.

``l``
  Optional.  Specifies that the name of the package information file on disk
  should be lower case.  See the ``LOWER_CASE_FILE`` option of
  :command:`install(PACKAGE_INFO)`.

``a<appendix-name>``
  Optional.  Specifies that an appendix ``<appendix-name>`` should be created
  rather than a root package description.  See the ``APPENDIX`` option of
  :command:`install(PACKAGE_INFO)`.  Note that additional information
  (see below) cannot be added to appendices.

``<destination>``
  Optional.  Specifies the destination to which the package information file
  should be installed.  See the ``DESTINATION`` option of
  :command:`install(PACKAGE_INFO)`.  Note that the default is a
  platform-specific location that is appropriate for |CPS| files in most
  instances, *not* the ``DESTINATION`` of the :command:`install(EXPORT)`
  which the directive matched.

For non-appendices, CMake will also infer additional information from several
CMake variables of the form ``<export-name>_EXPORT_PACKAGE_INFO_<var>``.  The
values of these are first processed as if by :command:`string(CONFIGURE)` with
the ``@ONLY`` option.  These are optional, and their effect is equivalent to
passing their value to the ``<var>`` option of the
:command:`install(PACKAGE_INFO)` command.

The additional variables are:

  * ``<export-name>_EXPORT_PACKAGE_INFO_VERSION``
  * ``<export-name>_EXPORT_PACKAGE_INFO_COMPAT_VERSION``
  * ``<export-name>_EXPORT_PACKAGE_INFO_VERSION_SCHEMA``
  * ``<export-name>_EXPORT_PACKAGE_INFO_LICENSE``
  * ``<export-name>_EXPORT_PACKAGE_INFO_DEFAULT_LICENSE``
  * ``<export-name>_EXPORT_PACKAGE_INFO_DEFAULT_CONFIGURATIONS``

Ideally, the version should be set to ``@PROJECT_VERSION@``.  However,
some projects may not use the ``VERSION`` option of the :command:`project`
command.

Example
^^^^^^^

Consider the following (simplified) project:

.. code-block:: cmake

  project(Example VERSION 1.2.3 SPDX_LICENSE "BSD-3-Clause")

  add_library(foo ...)
  add_library(bar ...)

  install(TARGETS foo EXPORT required-targets)
  install(TARGETS bar EXPORT optional-targets)

  install(EXPORT required-targets FILE example-targets.cmake ...)
  install(EXPORT optional-targets FILE example-optional-targets.cmake ...)

In this example, let ``example-targets.cmake`` be a file which is always
installed, and ``example-optional-targets.cmake`` be a file which is
optionally installed (e.g. is distributed as part of a separate package which
depends on the 'base' package).

Now, imagine we are a distributor that wants to make |CPS| package information
files available to our users, but we do not want to modify the project's build
files.  We can do this by passing the following arguments to CMake:

.. code-block::

  -DCMAKE_EXPERIMENTAL_FIXME=<elided>
  -DCMAKE_INSTALL_EXPORTS_AS_PACKAGE_INFO=\
    required-targets:Example/l;\
    optional-targets:Example/laoptional
  -Drequired-targets_EXPORT_PACKAGE_INFO_VERSION=@Example_VERSION@
  -Drequired-targets_EXPORT_PACKAGE_INFO_LICENSE=@Example_SPDX_LICENSE@

(Whitespace and line continuation characters, added for readability, should
be removed in real usage. Arguments may need to be quoted to prevent being
reinterpreted by the command shell.)

This will cause CMake to also create and install the files ``example.cps`` and
``example-optional.cps`` which describe the ``Example`` package.  We could
also specify the version and license information using substitutions provided
by the package build system (e.g. rpm, dpkg) if the project does not provide
this information via the :command:`project` command.

.. _CPS: https://cps-org.github.io/cps/
.. |CPS| replace:: Common Package Specification
