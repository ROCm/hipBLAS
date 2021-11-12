***********************
Building and Installing
***********************

Prerequisites
=============

-  A ROCm enabled platform, more information `here <https://rocm.github.io/>`_.

Installing pre-built packages
=============================

Download pre-built packages either from `ROCm's package servers <https://rocm.github.io/install.html#installing-from-amd-rocm-repositories>`_ or by clicking the GitHub releases tab and manually downloading, which could be newer.  Release notes are available for each release on the releases tab.

.. code-block::bash
   sudo apt update && sudo apt install hipblas

Quickstart hipBLAS build
========================

Build library dependencies + library
------------------------------------
The root of this repository has a helper bash script `install.sh` to build and install hipBLAS on Ubuntu with a single command.  It does take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.
A few commands in the script need sudo access so that it may prompt you for a password.

Typical uses of install.sh to build (library dependencies + library) are
in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
|  Command                                  | Description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -d``                       | Build library            |
|                                           | dependencies and library |
|                                           | in your local directory. |
|                                           | The -d flag only needs   |
|                                           | to be used once. For     |
|                                           | subsequent invocations   |
|                                           | of install.sh it is not  |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh``                          | Build library in your    |
|                                           | local directory. It is   |
|                                           | assumed dependencies     |
|                                           | have been built.         |
+-------------------------------------------+--------------------------+
| ``./install.sh -i``                       | Build library, then      |
|                                           | build and install        |
|                                           | hipBLAS package in       |
|                                           | /opt/rocm/hipblas. You   |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | hipBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+


Build library dependencies + client dependencies + library + client
-------------------------------------------------------------------

The client contains executables in the table below.

=============== ====================================================
executable name description
=============== ====================================================
hipblas-test    runs Google Tests to test the library
hipblas-bench   executable to benchmark or test individual functions
example-sscal   example C code calling hipblas_sscal function
=============== ====================================================

Common uses of install.sh to build (dependencies + library + client) are
in the table below.

.. tabularcolumns::
   |\X{2}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
| Command                                   | Description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -dc``                      | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | and client in your local |
|                                           | directory. The -d flag   |
|                                           | only needs to be used    |
|                                           | once. For subsequent     |
|                                           | invocations of           |
|                                           | install.sh it is not     |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh -c``                       | Build library and client |
|                                           | in your local directory. |
|                                           | It is assumed the        |
|                                           | dependencies have been   |
|                                           | built.                   |
+-------------------------------------------+--------------------------+
| ``./install.sh -idc``                     | Build library            |
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
| ``./install.sh -ic``                      | Build and install        |
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


Dependencies
============

Dependencies are listed in the script install.sh. The -d flag to install.sh installs dependencies.

CMake has a minimum version requirement listed in the file install.sh. See --cmake_install flag in install.sh to upgrade automatically.

