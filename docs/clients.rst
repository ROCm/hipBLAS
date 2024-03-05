.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _hipblas-clients:

********************************************************************
Clients
********************************************************************

There are two client executables that can be used with hipBLAS. They are,

1. hipblas-bench
2. hipblas-test

These two clients can be built by following the instructions in :doc:`./install`. After building the hipBLAS clients, they can be found in the directory ``hipBLAS/build/release/clients/staging``.

The next two sections will cover a brief explanation and the usage of each hipBLAS client.

hipblas-bench
=============

hipblas-bench is used to measure performance and to verify the correctness of hipBLAS functions.

It has a command line interface. For more information:

.. code-block:: bash

   ./hipblas-bench --help

For example, to measure performance of sgemm:

.. code-block:: bash

   ./hipblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096

On a vega20 machine the above command outputs a performance of 11941.5 Gflops below:

.. code-block:: bash

   transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,hipblas-Gflops,us
   N,N,4096,4096,4096,1,4096,4096,0,4096,11941.5,11509.4

A useful way of finding the parameters that can be used with ``./hipblas-bench -f gemm`` is to turn on logging
by setting environment variable ``ROCBLAS_LAYER=2``. For example if the user runs:

.. code-block:: bash

   ROCBLAS_LAYER=2 ./hipblas-bench -f gemm -i 1 -j 0

The above command will log:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 128 -n 128 -k 128 --alpha 1 --lda 128 --ldb 128 --beta 0 --ldc 128

The user can copy and change the above command. For example, to change the datatype to IEEE-64 bit and the size to 2048:

.. code-block:: bash

   ./hipblas-bench -f gemm -r f64_r --transposeA N --transposeB N -m 2048 -n 2048 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048


Logging affects performance, so only use it to log the command to copy and change, then run the command without logging to measure performance.

Note that hipblas-bench also has the flag ``-v 1`` for correctness checks.

If multiple arguments or even multiple functions need to be benchmarked there is support for data driven benchmarks via a yaml format specification file.

.. code-block:: bash

   ./hipblas-bench --yaml <file>.yaml

An example yaml file that is used for a smoke test is hipblas_smoke.yaml but other examples can be found in the rocBLAS repository.


hipblas-test
============

hipblas-test is used in performing hipBLAS unit tests and it uses Googletest framework.

To run the hipblas tests:

.. code-block:: bash

   ./hipblas-test

To run a subset of tests a filter may be provided. For example to only run axpy function tests via command line use:

.. code-block:: bash

   ./hibblas-test --gtest_filter=*axpy*

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]

If specific function arguments or even multiple functions need to be tested there is support for data driven testing via a yaml format test specification file.

.. code-block:: bash

   ./hipblas-test --yaml <file>.yaml

An example yaml file that is used to define a smoke test is hipblas_smoke.yaml but other examples can be found in the rocBLAS repository.  Yaml based
tests list function parameter values in the test name which can be also used for test filtering via the gtest_filter argument.
To run the provided smoke test use:

.. code-block:: bash

   ./hipblas-test --yaml hipblas_smoke.yaml
