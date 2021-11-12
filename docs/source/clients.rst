============
Clients
============

There are two client executables that can be used with hipBLAS. They are,

1. hipblas-bench

2. hipblas-test

These two clients can be built by following the instructions at `Building and Installing hipBLAS github page <https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/docs/source/install.rst>`_. After building the hipBLAS clients, they can be found in the directory ``hipBLAS/build/release/clients/staging``.

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

hipblas-test
============

hipblas-test is used in performing hipBLAS unit tests and it uses Googletest framework.

The tests are in 4 categories:

- quick
- pre_checkin
- nightly
- known_bug

To run the quick tests:

.. code-block:: bash

   ./hipblas-test --gtest_filter=*quick*

The other tests can also be run using the above command by replacing ``*quick*`` with ``*pre_checkin*``, ``*nightly*``, and ``*known_bug*``.

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]

gtest_filter can also be used to run tests for a particular function, and a particular set of input parameters. For example, to run all quick tests for the function hipblas_saxpy:

.. code-block:: bash

   ./hipblas-test --gtest_filter=*quick*axpy*f32_r*

The number of lines of output can be reduced with:

.. code-block:: bash

   GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_filter=*quick*


CUDA unit test failures
-----------------------
There are a few library unit tests failing with cuBLAS; we believe these failures are benign and can be ignored. Our unit tests are testing with negative strides and edge cases which are handled differently between the two libraries, and our unit tests don't account for these differences yet. These errors will be resolved in an upcoming release.

