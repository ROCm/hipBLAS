.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _hipblas-orga:

**********************************
Library Source Code Organization
**********************************

The hipBLAS code is split into two major parts:

- The ``library`` directory contains all source code for the library.
- The ``clients`` directory contains all test code and code to build clients.
- Infrastructure

The ``library`` directory
--------------------------

library/include
```````````````
Contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

library/src/amd_detail
```````````````````````
Implementation of hipBLAS interface compatible with rocBLAS APIs.

library/src/nvidia_detail
`````````````````````````
Implementation of hipBLAS interface compatible with cuBLAS-v2 APIs.

library/src/include
```````````````````
Internal include files for:

- Converting C++ exceptions to hipBLAS status.

The ``clients`` directory
-----------------------

clients/gtest
`````````````
Code for client hipblas-test. This client is used to test hipBLAS. Refer to :ref:`hipblas-clients` for more information. 

clients/benchmarks
``````````````````
Code for client hipblas-bench. This client is used to benchmark hipBLAS functions. Refer to :ref:`hipblas-clients` for more information. 

clients/include
```````````````
Code for testing and benchmarking individual hipBLAS functions, and utility code for testing.

clients/common
``````````````
Common code used by both hipblas-bench and hipblas-test.

clients/samples
```````````````
Sample code for calling hipBLAS functions.


Infrastructure
--------------

- CMake is used to build and package hipBLAS. There are ``CMakeLists.txt`` files throughout the code.
- Doxygen/Breathe/Sphinx/ReadTheDocs are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory ``library/include``
  - files in the directory ``docs``.

- Jenkins is used to automate Continuous Integration testing.
- clang-format is used to format C++ code.


