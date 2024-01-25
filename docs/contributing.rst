.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _contribute:

********************************************************************
Contributing to hipBLAS
********************************************************************

Pull-request guidelines
=======================


Our code contriubtion guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
The hipBLAS repository follows a workflow which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code. Pull requests should:

-  target the **develop** branch for integration
-  ensure code builds successfully.
-  do not break existing test cases
-  new unit tests should integrate within the existing googletest framework.
-  tests must have good code coverage
-  code must also have benchmark tests, and performance must approach
   the compute bound limit or memory bound limit.

Coding Guidelines:
==================
-  Do not use unnamed namespaces inside of header files.

-  Use either ``template`` or ``inline`` (or both) for functions defined outside of classes in header files.

-  Do not declare namespace-scope (not ``class``-scope) functions ``static`` inside of header files unless there is a very good reason, that the function does not have any non-``const`` ``static`` local variables, and that it is acceptable that each compilation unit will have its own independent definition of the function and its ``static`` local variables. (``static`` ``class`` member functions defined in header files are okay.)

-  Use ``static`` for ``constexpr`` ``template`` variables until C++17, after which ``constexpr`` variables become ``inline`` variables, and thus can be defined in multiple compilation units. It is okay if the ``constexpr`` variables remain ``static`` in C++17; it just means there might be a little bit of redundancy between compilation units.

Format
------

C and C++ code is formatted using ``clang-format``. To run clang-format
use the version in the ``/opt/rocm/llvm/bin`` directory. Please do not use your
system's built-in ``clang-format``, as this may be an older version that
will result in different results.

To format a file, use:

::

    /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install

Static Code Analysis
=====================

``cppcheck`` is an open-source static analysis tool. This project uses this tool for performing static code analysis.

Users can use the following command to run cppcheck locally to generate the report for all files.

.. code:: bash

   $ cd hipBLAS
   $ cppcheck --enable=all --inconclusive --library=googletest --inline-suppr -i./build --suppressions-list=./CppCheckSuppressions.txt --template="{file}:{line}: {severity}: {id} :{message}" . 2> cppcheck_report.txt

For more information on the command line options, refer to the cppcheck manual on the web.
