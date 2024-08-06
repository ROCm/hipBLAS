.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _windows-install:

***********************
Installation and Building for Windows
***********************

Prerequisites
=============

* If using the rocBLAS backend on an AMD machine:

  * An AMD HIP SDK enabled platform. Find more information on the :doc:`System requirements (Windows) <rocm-install-on-windows:reference/system-requirements>` page.
  * A compatible version of rocBLAS.
  * A compatible version of rocSOLVER and its dependencies, rocSPARSE and rocPRIM, for full functionality.
  * hipBLAS is supported on the same Windows versions and toolchains that are supported by the HIP SDK.

* hipBLAS does not currently support using the cuBLAS backend on Windows.

Installing pre-built packages
=============================

hipBLAS can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use hipBLAS in your code would be using CMake for which you would add the SDK installation location to your
``CMAKE_PREFIX_PATH`` in your CMake configure step.

::

    -DCMAKE_PREFIX_PATH="C:\hipSDK"


Then in your ``CMakeLists.txt`` use:

::

    find_package(hipblas)
    target_link_libraries( your_exe PRIVATE roc::hipblas )

Building and Installing hipBLAS
===============================

For most users, building from source is not necessary, as hipBLAS can be used after installing the pre-built packages as described above. However, users can use the following instructions to build hipBLAS from source if necessary.

Download hipBLAS Source Code
----------------------------

The hipBLAS source code is available at the `hipBLAS github page <https://github.com/ROCm/hipBLAS>`_. The version of the ROCm HIP SDK may be shown in the path of default installation, but you can run the HIP SDK compiler to report the versino from the bin/ folder with:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, HIP version could be 5.4.22880-135e1ab4;
this corresponds to major = 5, minor = 4, patch = 22880, build identifier 135e1ab4.
There are GitHub branches at the hipBLAS site with names release/rocm-rel-major.minor where major and minor are the same as in the HIP version.
For example for you can use the following to download rocBLAS:

::

git clone -b release/rocm-rel-x.y https://github.com/ROCm/hipBLAS.git

Replace x.y in the above command with the version of the HIP SDK installed on your machine. For example, if you have HIP 6.2 installed, then use ``-b release/rocm-rel-6.2``. You can add the SDK tools to your path with an entry such as:

::

    %HIP_PATH%\bin

Building
--------

The root of this repository has a helper python script ``rmake.py`` to build and install hipBLAS with a single command. It does take a lot of options and hard-coded configuration that can be specified through invoking ``cmake`` directly, but it's a great way to get started quickly and can serve as an example of how to build and install.
A few commands in the script need sudo access so it may prompt you for a password.

Typical uses of ``rmake.py`` to build (library dependencies + library) are
in the table below.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
|  Command                                  | Description              |
+===========================================+==========================+
| ``python3 rmake.py -h``                   | Help information.        |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -d``                   | Build library            |
|                                           | dependencies and library |
|                                           | in your local directory. |
|                                           | The -d flag only needs   |
|                                           | to be used once. For     |
|                                           | subsequent invocations   |
|                                           | of rmake.py it is not    |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py``                      | Build library in your    |
|                                           | local directory. It is   |
|                                           | assumed dependencies     |
|                                           | have been built.         |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -i``                   | Build library, then      |
|                                           | build and install        |
|                                           | hipBLAS package in       |
|                                           | `C:\\hipSDK`. You        |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | hipBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -n``                   | Build library without    |
|                                           | functionality provided   |
|                                           | by rocSOLVER.            |
|                                           | rocSOLVER, rocSPARSE,    |
|                                           | and rocPRIM dependencies |
|                                           | will not be needed.      |
|                                           | This flag has no effect  |
|                                           | when building with cuda  |
|                                           | backend.                 |
+-------------------------------------------+--------------------------+


Build library dependencies + client dependencies + library + client
-------------------------------------------------------------------

The client contains executables in the table below.

================= ====================================================
Executable name   Description
================= ====================================================
hipblas-test      runs Google Tests to test the library
hipblas-bench     executable to benchmark or test individual functions
hipblas-example-* various examples showing hipblas usage
================= ====================================================

Common uses of ``rmake.py`` to build (dependencies + library + client) are
in the table below.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
| Command                                   | Description              |
+===========================================+==========================+
| ``python3 rmake.py -dc``                  | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | and client in your local |
|                                           | directory. The -d flag   |
|                                           | only needs to be used    |
|                                           | once. For subsequent     |
|                                           | invocations of           |
|                                           | rmake.py it is not       |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -c``                   | Build library and client |
|                                           | in your local directory. |
|                                           | It is assumed the        |
|                                           | dependencies have been   |
|                                           | built.                   |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -idc``                 | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | client, then build and   |
|                                           | install the hipBLAS      |
|                                           | package. You will be     |
|                                           | prompted for sudo        |
|                                           | access. It is expected   |
|                                           | that if you want to      |
|                                           | install for all users    |
|                                           | you use the -i flag. If  |
|                                           | you want to keep hipBLAS |
|                                           | in your local directory, |
|                                           | you do not need the -i   |
|                                           | flag.                    |
+-------------------------------------------+--------------------------+
| ``python3 rmake.py -ic``                  | Build and install        |
|                                           | hipBLAS package, and     |
|                                           | build the client. You    |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | hipBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+

Dependencies for building library
==================================

Use ``rmake.py`` with ``-d`` option to install dependencies required to build the library. This will not install the rocBLAS, rocSOLVER, rocSPARSE, and rocPRIM dependencies.
When building hipBLAS it is important to note version dependencies of other libraries. The rocBLAS and rocSOLVER versions needed for an AMD backend build are listed in the top level CMakeLists.txt file.
rocSPARSE and rocPRIM are currently dependencies of rocSOLVER. To build these libraries from source, please visit the :doc:`rocBLAS Documentation <rocBLAS:index>`,
:doc:`rocSOLVER Documentation <rocSOLVER:index>`, :doc:`rocSPARSE Documentation <rocSPARSE:index>`, and :doc:`rocPRIM Documentation <rocPRIM:index>`.

CMake has a minimum version requirement which is currently 3.16.8. See ``--cmake_install`` flag in ``rmake.py`` to upgrade automatically.

For the test and benchmark clients' host reference functions you must manually download and install AMD's ILP64 version of the AOCL libraries, version 4.2, from https://www.amd.com/en/developer/aocl.html.
If you download and run the full Windows AOCL installer into the default location (``C:\Program Files\AMD\AOCL-Windows\``) then the AOCL reference BLAS (amd-blis) should be found by the clients' CMakeLists.txt.

Note, if you only use the ``rmake.py -d`` dependency script and change the default CMake option ``LINK_BLIS=ON``, you may experience `hipblas-test` stress test failures due to 32-bit integer overflow
on the host unless you exclude the stress tests via command line argument ``--gtest_filter=-*stress*``.
