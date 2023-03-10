***********************
API
***********************

hipBLAS interface examples
---------------------------

The hipBLAS interface is compatible with rocBLAS and cuBLAS-v2 APIs.  Porting a CUDA application which originally calls the cuBLAS API to an application calling hipBLAS API should be relatively straightforward. For example, the hipBLAS SGEMV interface is

GEMV API
--------

.. code-block:: cpp

   hipblasStatus_t
   hipblasSgemv( hipblasHandle_t handle,
               hipblasOperation_t trans,
               int m, int n, const float *alpha,
               const float *A, int lda,
               const float *x, int incx, const float *beta,
               float *y, int incy );


Batched and strided GEMM API
-----------------------------

hipBLAS GEMM can process matrices in batches with regular strides.  There are several permutations of these API's, the
following is an example that takes everything

.. code-block:: cpp

   hipblasStatus_t
   hipblasSgemmStridedBatched( hipblasHandle_t handle,
               hipblasOperation_t transa, hipblasOperation_t transb,
               int m, int n, int k, const float *alpha,
               const float *A, int lda, long long bsa,
               const float *B, int ldb, long long bsb, const float *beta,
               float *C, int ldc, long long bsc,
               int batchCount);

hipBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data.  Users are
responsible for copying data from/to the host and device memory.
