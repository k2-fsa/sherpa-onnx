.. cmake-manual-description: CMake Instrumentation

cmake-instrumentation(7)
************************

.. versionadded:: 4.3

.. only:: html

  .. contents::

Introduction
============

The CMake Instrumentation API allows for the collection of timing data, target
information and system diagnostic information during the configure, generate,
build, test and install steps for a CMake project.

All interactions with the CMake instrumentation API must specify both an API
version and a Data version. At this time, there is only one version for each of
these: the `API v1`_ and `Data v1`_.

.. note::

  This feature is only available for projects using the
  :ref:`Makefile Generators`, :ref:`Ninja Generators` or :generator:`FASTBuild`.

Overview
--------

CMake Instrumentation works in 2 major stages: `Data Collection`_, and
`Indexing`_.

`Data Collection`_ is the process by which CMake writes out instrumentation
data into a project build tree. The commands instrumented in this way range
from overall builds, to individual compile commands. The full list of commands
instrumented is documented under the `v1 Snippet File`_.

Collected data will accumulate until `Indexing`_ occurs: the process of
collating the generated data. Indexing occurs on events called "hooks",
which can be configured as part of the `v1 Query Files`_. A `v1 Index File`_ is
created and passed to any user-defined `Callbacks`_ provided to process
the data. Once all `Callbacks`_ have run, CMake deletes the files.

Without the need for any custom `Callbacks`_, CMake can submit instrumentation
data to `CDash`_, or generate a `Google Trace File`_ for visualization.

Data Collection
---------------

Whenever a command is executed with instrumentation enabled, a
`v1 Snippet File`_ is created in the project build tree with data specific to
that command. These files remain until after `Indexing`_ occurs.

CMake sets the :prop_gbl:`RULE_LAUNCH_COMPILE`, :prop_gbl:`RULE_LAUNCH_LINK`
and :prop_gbl:`RULE_LAUNCH_CUSTOM` global properties to wrap each compile, link
and custom command invocation in a launcher that performs instrumentation and
writes out a `v1 Snippet File`_. If the project has been configured with
:module:`CTestUseLaunchers`, the launcher will collect instrumentation data in
addition to performing the communication typically handled by that module.

.. _`cmake-instrumentation Indexing`:

Indexing
--------

Indexing is the process of collating generated instrumentation data. The
available hooks to trigger indexing include options such as after every build,
or every :manual:`ctest <ctest(1)>` invocation, and are configured as part of the `v1 Query Files`_.
Whenever a hook is triggered, an index file is generated containing a list of
snippet files newer than the previous indexing. This index file is passed to
user-defined `Callbacks`_ commands to process the data.

Indexing and can also be performed by manually invoking
:option:`ctest --collect-instrumentation`.

.. _`cmake-instrumentation Callbacks`:

Callbacks
^^^^^^^^^

As part of the `v1 Query Files`_, users can provide a list of callbacks
intended to handle data collected by this feature.

Whenever `Indexing`_ occurs, each provided callback is executed, passing the
path to the generated `v1 Index File`_ as an additional argument.

These callbacks, defined either at the user-level or project-level, should read
the instrumentation data and perform any desired handling of it. The index file
and its listed snippets are automatically deleted by CMake once all callbacks
have completed. Note that a callback should never move or delete these data
files manually as they may be needed by other callbacks.

If indexing is triggered again before `Callbacks`_ have finished running,
the generated index file will contain only instrumentation data generated since
the previous indexing.

Enabling Instrumentation
========================

Instrumentation can be enabled either for an individual CMake project, or
for all CMake projects configured and built by a user. In all cases, a "query"
represents a request for instrumentation behavior. See the `v1 Query Files`_
for details on configuring this feature.

Enabling Instrumentation at the Project-Level
---------------------------------------------

Project code can contain instrumentation queries by using the
:command:`cmake_instrumentation` command.

In addition, query files can be placed manually under
``<build>/.cmake/instrumentation/<version>/query/`` at the top of a build tree.
This version of CMake supports only one version schema, `API v1`_.

Enabling Instrumentation at the User-Level
------------------------------------------

Instrumentation can be configured at the user-level by placing query files in
the :envvar:`CMAKE_CONFIG_DIR` under
``<config_dir>/instrumentation/<version>/query/``.

.. _`CDash`:

Enabling Instrumentation for CDash Submissions
----------------------------------------------

You can enable instrumentation when using :module:`CTest` in
:ref:`Dashboard Client` mode by setting the :envvar:`CTEST_USE_INSTRUMENTATION`
environment variable. Doing so automatically enables the
``dynamicSystemInformation`` option.

The following table shows how each type of instrumented command gets mapped
to a corresponding type of CTest XML file.

=================================================== ==================
:ref:`Snippet Role <cmake-instrumentation Data v1>` CTest XML File
=================================================== ==================
``configure``                                       ``Configure.xml``
``generate``                                        ``Configure.xml``
``compile``                                         ``Build.xml``
``link``                                            ``Build.xml``
``custom``                                          ``Build.xml``
``build``                                           unused!
``cmakeBuild``                                      ``Build.xml``
``cmakeInstall``                                    ``Build.xml``
``install``                                         ``Build.xml``
``ctest``                                           ``Build.xml``
``test``                                            ``Test.xml``
=================================================== ==================

By default the command line reported to CDash is truncated at the first space.
You can instead choose to report the full command line (including arguments)
by setting :envvar:`CTEST_USE_VERBOSE_INSTRUMENTATION` to 1.

Alternatively, you can use the `v1 Query Files`_ to enable instrumentation for
CDash using the ``cdashSubmit`` and ``cdashVerbose`` options.

In order for the submitted ``Build.xml`` file to group the snippet files
correctly, all configure and build commands should be executed with CTest in
Dashboard Client mode.

.. _`cmake-instrumentation API v1`:

API v1
======

The API version specifies the layout of the instrumentation directory, as well
as the general format of the query files and :command:`cmake_instrumentation`
command arguments.

The Instrumentation API v1 is housed  in the ``instrumentation/v1/`` directory
under either ``<build>/.cmake/`` for output data and project-level queries, or
``<config_dir>/`` for user-level queries. The ``v1`` component of this
directory is what signifies the API version. It has the following
subdirectories:

``query/``
  Holds query files written by users or clients. Any file with the ``.json``
  file extension will be recognized as a query file. These files are owned by
  whichever client or user creates them.

``query/generated/``
  Holds query files generated by a CMake project with the
  :command:`cmake_instrumentation` command or the
  :envvar:`CTEST_USE_INSTRUMENTATION` variable. These files are owned by CMake
  and are deleted and regenerated automatically during the CMake configure
  step.

``data/``
  Holds instrumentation data collected on the project. CMake owns all data
  files, they should never be removed by other processes. Data collected here
  remains until after `Indexing`_ occurs and all `Callbacks`_ are executed.

``data/index/``
  A subset of the collected data, containing any
  `v1 Index Files <v1 Index File_>`_.

``data/content/``
  A subset of the collected data, containing any
  `v1 CMake Content Files <v1 CMake Content File_>`_.

``data/trace/``
  A subset of the collected data, containing the `Google Trace File`_ created
  from the most recent `Indexing`_. Unlike other data files, the most recent
  trace file remains even after `Indexing`_ occurs and all `Callbacks`_ are
  executed, until the next time `Indexing`_ occurs.

``cdash/``
  Holds temporary files used internally to generate XML content to be submitted
  to CDash.

.. _`cmake-instrumentation v1 Query Files`:

v1 Query Files
--------------

Any file with the ``.json`` extension under the ``instrumentation/v1/query/``
directory is recognized as a query for instrumentation data.

These files must contain a JSON object with the following keys. The ``version``
key is required, but all other fields are optional.

``version``
  The Data version of snippet file to generate, an integer. Currently the only
  supported version is ``1``.

``callbacks``
  A list of command-line strings for `Callbacks`_ to handle collected
  instrumentation data. Whenever these callbacks are executed, the full path to
  a `v1 Index File`_ is appended to the arguments included in the string.

``hooks``
  A list of strings specifying when `Indexing`_ should occur automatically.
  These are the intervals when instrumentation data should be collated and user
  `Callbacks`_ should be invoked to handle the data. Elements in this list
  should be one of the following:

  * ``postGenerate``
  * ``preBuild`` (called when ``ninja``  or ``make`` is invoked)
  * ``postBuild`` (called when ``ninja`` or ``make`` completes)
  * ``preCMakeBuild`` (called when :option:`cmake --build` is invoked)
  * ``postCMakeBuild`` (called when :option:`cmake --build` completes)
  * ``postCMakeInstall``
  * ``postCMakeWorkflow``
  * ``postCTest``

  ``preBuild`` and ``postBuild`` are not supported when using the
  :generator:`MSYS Makefiles` or :generator:`FASTBuild` generators.
  Additionally, they will not be triggered when the build tool is invoked by
  :option:`cmake --build`.

``options``
  A list of strings used to enable certain optional behavior, including the
  collection of certain additional data. Elements in this list should be one of
  the following:

    ``staticSystemInformation``
      Enables collection of the static information about the host machine CMake
      is being run from. This data is collected during `Indexing`_ and is
      included in the generated `v1 Index File`_.

    ``dynamicSystemInformation``
      Enables collection of the dynamic information about the host machine
      CMake is being run from. Data is collected for every `v1 Snippet File`_
      generated by CMake, and includes information from immediately before and
      after the command is executed.

    ``cdashSubmit``
      Enables including instrumentation data in CDash. This is
      equivalent to having the :envvar:`CTEST_USE_INSTRUMENTATION` environment
      variable enabled.

    ``cdashVerbose``
      Enables including the full untruncated commands in data submitted to
      CDash. Equivalent to having the
      :envvar:`CTEST_USE_VERBOSE_INSTRUMENTATION` and
      :envvar:`CTEST_USE_INSTRUMENTATION` environment variables enabled.

    ``trace``
      Enables generation of a `Google Trace File`_ during `Indexing`_ to
      visualize data from the `v1 Snippet Files <v1 Snippet File_>`_ collected.

The ``callbacks`` listed will be invoked during the specified hooks
*at a minimum*. When there are multiple query files, the ``callbacks``,
``hooks`` and ``options`` between them will be merged. Therefore, if any query
file includes any ``hooks``, every ``callback`` across all query files will be
executed at every ``hook`` across all query files. Additionally, if any query
file requests optional data using the ``options`` field, any related data will
be present in all snippet files. User written ``callbacks`` should be able to
handle the presence of this optional data, since it may be requested by an
unrelated query.

The JSON format is described in machine-readable form by
:download:`this JSON schema </manual/instrumentation/query-v1-schema.json>`.

Example:

.. code-block:: json

  {
    "version": 1,
    "callbacks": [
      "/usr/bin/python callback.py",
      "/usr/bin/cmake -P callback.cmake arg",
    ],
    "hooks": [
      "postCMakeBuild",
      "postCMakeInstall"
    ],
    "options": [
      "staticSystemInformation",
      "dynamicSystemInformation",
      "cdashSubmit",
      "trace"
    ]
  }

In this example, after every :option:`cmake --build` or
:option:`cmake --install` invocation, an index file ``index-<timestamp>.json``
will be generated in ``<build>/.cmake/instrumentation/v1/data/index``
containing a list of data snippet files created since the previous indexing.
The commands ``/usr/bin/python callback.py index-<timestamp>.json`` and
``/usr/bin/cmake -P callback.cmake arg index-<timestamp>.json`` will be
executed in that order. The index file will contain the
``staticSystemInformation`` data and each snippet file listed in the index will
contain the ``dynamicSystemInformation`` data. Additionally, the index file
will contain the path to the generated `Google Trace File`_. Once both
callbacks have completed, the index file and data files listed by it (including
snippet files, but not the trace file) will be deleted from the project build
tree. The instrumentation data will be present in the XML files submitted to
CDash, but with truncated command strings because ``cdashVerbose`` was not
enabled.

.. _`cmake-instrumentation Data v1`:

Data v1
=======

Data version specifies the contents of the output files generated by the CMake
instrumentation API as part of the `Data Collection`_ and `Indexing`_. A new
version number will be created whenever previously included data is removed or
reformatted such that scripts written to parse this data may become
incompatible with the new format. There are four types of data files generated:
the `v1 Snippet File`_, `v1 Index File`_, `v1 CMake Content File`_, and the
`Google Trace File`_. When using the `API v1`_, these files live in
``<build>/.cmake/instrumentation/v1/data/`` under the project build tree.

.. _`cmake-instrumentation v1 Snippet File`:

v1 Snippet File
---------------

Snippet files are generated for every compile, link and custom command invoked
as part of the CMake build or install step and contain instrumentation data
about the command executed. Additionally, snippet files are created for the
following:

* The CMake configure step
* The CMake generate step
* Entire build step (executed with :option:`cmake --build`)
* Entire install step (executed with :option:`cmake --install`)
* Each time :manual:`ctest <ctest(1)>` is invoked to
  :ref:`run tests <Run Tests>` (even if no tests are found)
* Each individual test executed by :manual:`ctest <ctest(1)>`

These files remain in the build tree until after `Indexing`_ occurs and any
user-specified `Callbacks`_ are executed.

.. note::

  Configure and generate snippet files are not written by CMake until the
  generate step is complete. When using :manual:`cmake-gui(1)` or
  :manual:`ccmake(1)`, triggering only configure step(s) without generating the
  project files will not generate any configure snippets. Once the generate
  step is run, there will be one configure snippet for each time the configure
  step was run.

Snippet files have a filename with the syntax
``<role>-<hash>-<timestamp>.json`` and contain the following data:

  ``version``
    The Data version of the snippet file, an integer. Currently the version is
    always ``1``.

  ``command``
    The full command executed. Excluded when ``role`` is ``build``.

  ``workingDir``
    The working directory in which the ``command`` was executed.

  ``result``
    The exit code of the command, an integer. This will be ``null`` when
    ``role`` is ``build``.

  ``role``
    The type of command executed, which will be one of the following values:

    * ``configure``: the CMake configure step
    * ``generate``: the CMake generate step
    * ``compile``: an individual compile step invoked during the build
    * ``link``: an individual link step invoked during the build
    * ``custom``: an individual custom command invoked during the build
    * ``build``: a complete ``make`` or ``ninja`` invocation
      (not through :option:`cmake --build`).
    * ``cmakeBuild``: a complete :option:`cmake --build` invocation
    * ``cmakeInstall``: a complete :option:`cmake --install` invocation
    * ``install``: an individual ``cmake -P cmake_install.cmake`` invocation
    * ``ctest``: a complete :manual:`ctest <ctest(1)>` command invocation
    * ``test``: a single test executed by :manual:`ctest <ctest(1)>`

  ``target``
    The CMake target associated with the command. Only included when ``role``
    is ``compile`` or ``link``, or when ``role`` is ``custom`` and the custom
    command is attached to a target with :command:`add_custom_command(TARGET)`.
    In conjunction with ``cmakeContent``, this can be used to look up the
    target :prop_tgt:`TYPE` and :prop_tgt:`LABELS`.

  ``timeStart``
    Time at which the command started, expressed as the number of milliseconds
    since the system epoch.

  ``duration``
    The duration that the command ran for, expressed in milliseconds.

  ``outputs``
    The command's output file(s), an array. Only included when ``role`` is one
    of: ``compile``, ``link``, ``custom``.

  ``outputSizes``
    The size(s) in bytes of the ``outputs``, an array. For files which do not
    exist, the size is 0. Included under the same conditions as the ``outputs``
    field.

  ``source``
    The source file being compiled. Only included when ``role`` is ``compile``.

  ``language``
    The language of the source file being compiled. Only included when ``role``
    is ``compile`` or ``link``.

  ``testName``
    The name of the test being executed. Only included when ``role`` is
    ``test``.

  ``config``
    The :ref:`Build Configuration <Build Configurations>`, such as ``Release``
    or ``Debug``. Only included when ``role`` is one of: ``compile``, ``link``,
    ``custom``, ``install``, ``test``.

  ``dynamicSystemInformation``
    Specifies the dynamic information collected about the host machine
    CMake is being run from. Data is collected for every snippet file
    generated by CMake, with data immediately before and after the command is
    executed. Only included when enabled by the `v1 Query Files`_.

    ``beforeHostMemoryUsed``
      The Host Memory Used in KiB at ``timeStart``.

    ``afterHostMemoryUsed``
      The Host Memory Used in KiB at ``timeStart + duration``.

    ``beforeCPULoadAverage``
      The Average CPU Load at ``timeStart``, or ``null`` if it cannot be
      determined.

    ``afterCPULoadAverage``
      The Average CPU Load at ``timeStart + duration``, or ``null`` if it
      cannot be determined.

  ``cmakeContent``
    The path to a `v1 CMake Content File`_ located under ``data``, which
    contains information about the CMake configure and generate steps
    responsible for generating the ``command`` in this snippet. When using
    :manual:`cmake-gui(1)` or :manual:`ccmake(1)`, this field may be ``null``
    for all configure steps up to the most recent one before the generate step.

  ``showOnly``
    A boolean representing whether the
    :option:`--show-only <ctest --show-only>` option was passed to
    :manual:`ctest <ctest(1)>`. Only included when ``role`` is ``ctest``.

Example:

.. code-block:: json

  {
    "version": 1,
    "command" : "\"/usr/bin/c++\" \"-MD\" \"-MT\" \"CMakeFiles/main.dir/main.cxx.o\" \"-MF\" \"CMakeFiles/main.dir/main.cxx.o.d\" \"-o\" \"CMakeFiles/main.dir/main.cxx.o\" \"-c\" \"<src>/main.cxx\"",
    "role" : "compile",
    "result" : 1,
    "target": "main",
    "language" : "C++",
    "outputs" : [ "CMakeFiles/main.dir/main.cxx.o" ],
    "outputSizes" : [ 0 ],
    "source" : "<src>/main.cxx",
    "config" : "Debug",
    "dynamicSystemInformation" :
    {
      "afterCPULoadAverage" : 2.3500000000000001,
      "afterHostMemoryUsed" : 6635680.0
      "beforeCPULoadAverage" : 2.3500000000000001,
      "beforeHostMemoryUsed" : 6635832.0
    },
    "timeStart" : 1737053448177,
    "duration" : 31,
    "cmakeContent" : "content/cmake-2025-07-11T12-46-32-0572.json"
  }

v1 Index File
-------------

Index files contain a list of `v1 Snippet Files <v1 Snippet File_>`_. It
serves as an entry point for navigating the instrumentation data. They are
generated whenever `Indexing`_ occurs and deleted after any user-specified
`Callbacks`_ are executed.

``version``
  The Data version of the index file, an integer. Currently the version is
  always ``1``.

``buildDir``
  The build directory of the CMake project.

``dataDir``
  The full path to the ``<build>/.cmake/instrumentation/v1/data/`` directory.

``hook``
  The name of the hook responsible for generating the index file. In addition
  to the hooks that can be specified by one of the `v1 Query Files`_, this
  value may be set to ``manual`` if indexing is performed by invoking
  :option:`ctest --collect-instrumentation`.

  Note that the hook is not directly tied to what data may be available.
  A ``postBuild`` hook, for example, may include ``test`` or ``install``
  snippets, if these steps were run since the previous indexing.

``snippets``
  Contains a list of `v1 Snippet Files <v1 Snippet File_>`_. This includes all
  snippet files generated since the previous index file was created. The file
  paths are relative to ``dataDir``.

  This list may be empty if indexing was run twice in succession, such as when
  building multiple times with both the ``preBuild`` and ``postBuild`` hooks
  enabled.

``trace``
  Contains the path to the `Google Trace File`_. This includes data from all
  corresponding ``snippets`` in the index file. The file path is relative to
  ``dataDir``. Only included when enabled by the `v1 Query Files`_.

``staticSystemInformation``
  Specifies the static information collected about the host machine
  CMake is being run from. If CMake is unable to determine the value of any
  given field, it will be ``null``. See :command:`cmake_host_system_information`
  for a description of each of the following fields.

  Only included when enabled by the `v1 Query Files`_.

  * ``OSName``
  * ``OSPlatform``
  * ``OSRelease``
  * ``OSVersion``
  * ``familyId``
  * ``hostname``
  * ``is64Bits``
  * ``modelId``
  * ``modelName``
  * ``numberOfLogicalCPU``
  * ``numberOfPhysicalCPU``
  * ``processorAPICID``
  * ``processorCacheSize``
  * ``processorClockFrequency``
  * ``processorName``
  * ``totalPhysicalMemory``
  * ``totalVirtualMemory``
  * ``vendorID``
  * ``vendorString``

Example:

.. code-block:: json

  {
    "version": 1,
    "hook": "manual",
    "buildDir": "<build>",
    "dataDir": "<build>/.cmake/instrumentation/v1/data",
    "snippets": [
      "configure-<hash>-<timestamp>.json",
      "generate-<hash>-<timestamp>.json",
      "compile-<hash>-<timestamp>.json",
      "compile-<hash>-<timestamp>.json",
      "link-<hash>-<timestamp>.json",
      "install-<hash>-<timestamp>.json",
      "ctest-<hash>-<timestamp>.json",
      "test-<hash>-<timestamp>.json",
      "test-<hash>-<timestamp>.json",
    ],
    "trace": "trace/trace-<timestamp>.json"
  }

.. _`cmake-instrumentation v1 CMake Content File`:

v1 CMake Content File
---------------------

CMake content files contain information about the CMake configure and generate
steps. Each `v1 Snippet File`_ provides the path to one of these files
corresponding to the CMake invocation responsible for generating its command.

Each CMake content file contains the following:

  ``project``
    The value of :variable:`CMAKE_PROJECT_NAME`.

  ``custom``
    An object containing arbitrary JSON data specified by the user with the
    :ref:`cmake_instrumentation CUSTOM_CONTENT` functionality of the
    :command:`cmake_instrumentation` command.

  ``targets``
    An object containing CMake targets, indexed by name, that have
    corresponding instrumentation data. Each target contains the following:

    ``type``
      The :prop_tgt:`TYPE` property of the target. Only ``EXECUTABLE``,
      ``STATIC_LIBRARY``, ``SHARED_LIBRARY``, ``MODULE_LIBRARY`` and
      ``OBJECT_LIBRARY`` targets are included.

    ``labels``
      The :prop_tgt:`LABELS` property of the target.

.. _`cmake-instrumentation Google Trace File`:

Google Trace File
-----------------

CMake can generate a file in the `Google Trace Event Format`_ to help visualize
collected instrumentation data. Enabling the ``trace`` option in the
`v1 Query Files`_ causes such a file to be generated under
``<build>/.cmake/v1/instrumentation/data/trace`` whenever `Indexing`_ occurs.

Generated trace files include data from all
`v1 Snippet Files <v1 Snippet File_>`_ listed in the current index file.

When instrumentation data is deleted by CMake after `Indexing`_, the most
recent trace file remains so that it can be manually inspected without the need
for any custom `Callbacks`_.

Trace files are stored in the ``JSON Array Format``, where each
`v1 Snippet File`_ corresponds to a single trace event object. Each trace
event contains the following data:

``name``
  A descriptive name generated by CMake based on the given snippet data.

``cat``
  The ``role`` from the `v1 Snippet File`_.

``ph``
  Currently, always ``"X"`` to represent "Complete Events".

``ts``
  The ``timeStart`` from the `v1 Snippet File`_, converted from milliseconds to
  microseconds.

``dur``
  The ``duration`` from the `v1 Snippet File`_, converted from milliseconds to
  microseconds.

``pid``
  Unused (always zero).

``tid``
  An integer ranging from zero to the number of concurrent jobs with which the
  processes being indexed ran. This is a synthetic ID calculated by CMake
  based on the ``ts`` and ``dur`` of all snippet files being indexed in
  order to produce a more useful visualization of the process concurrency.

``args``
  Contains all data from the `v1 Snippet File`_ corresponding to this trace
  event.

.. _`Google Trace Event Format`: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
