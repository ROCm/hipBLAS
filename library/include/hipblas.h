/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//!  This is the master include file for hipblas, wrapping around rocblas and cublas "version 2"
//
#ifndef HIPBLAS_H
#define HIPBLAS_H
#pragma once
#include "hipblas-export.h"
#include "hipblas-version.h"
#include <hip/hip_runtime_api.h>

typedef void* hipblasHandle_t;

typedef uint16_t hipblasHalf;

enum hipblasStatus_t
{
    HIPBLAS_STATUS_SUCCESS           = 0, // Function succeeds
    HIPBLAS_STATUS_NOT_INITIALIZED   = 1, // HIPBLAS library not initialized
    HIPBLAS_STATUS_ALLOC_FAILED      = 2, // resource allocation failed
    HIPBLAS_STATUS_INVALID_VALUE     = 3, // unsupported numerical value was passed to function
    HIPBLAS_STATUS_MAPPING_ERROR     = 4, // access to GPU memory space failed
    HIPBLAS_STATUS_EXECUTION_FAILED  = 5, // GPU program failed to execute
    HIPBLAS_STATUS_INTERNAL_ERROR    = 6, // an internal HIPBLAS operation failed
    HIPBLAS_STATUS_NOT_SUPPORTED     = 7, // function not implemented
    HIPBLAS_STATUS_ARCH_MISMATCH     = 8,
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9 // hipBLAS handle is null pointer
};

// set the values of enum constants to be the same as those used in cblas
enum hipblasOperation_t
{
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113
};

enum hipblasPointerMode_t
{
    HIPBLAS_POINTER_MODE_HOST,
    HIPBLAS_POINTER_MODE_DEVICE
};

enum hipblasFillMode_t
{
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL  = 123
};

enum hipblasDiagType_t
{
    HIPBLAS_DIAG_NON_UNIT = 131,
    HIPBLAS_DIAG_UNIT     = 132
};

enum hipblasSideMode_t
{
    HIPBLAS_SIDE_LEFT  = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH  = 143
};

enum hipblasDatatype_t
{
    HIPBLAS_R_16F = 150,
    HIPBLAS_R_32F = 151,
    HIPBLAS_R_64F = 152,
    HIPBLAS_C_16F = 153,
    HIPBLAS_C_32F = 154,
    HIPBLAS_C_64F = 155
};

enum hipblasGemmAlgo_t
{
    HIPBLAS_GEMM_DEFAULT = 160
};

#ifdef __cplusplus
extern "C" {
#endif

HIPBLAS_EXPORT hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);

HIPBLAS_EXPORT hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);

HIPBLAS_EXPORT hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);

HIPBLAS_EXPORT hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId);

HIPBLAS_EXPORT hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t      handle,
                                                     hipblasPointerMode_t mode);

HIPBLAS_EXPORT hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t       handle,
                                                     hipblasPointerMode_t* mode);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasSgeam(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            const float*       beta,
                                            const float*       B,
                                            int                ldb,
                                            float*             C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeam(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            const double*      beta,
                                            const double*      B,
                                            int                ldb,
                                            double*            C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int
incx, float  *result, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int
incx, double *result, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            const float*    x,
                                            int             incx,
                                            float*          y,
                                            int             incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,
                                            int             n,
                                            const double*   alpha,
                                            const double*   x,
                                            int             incx,
                                            double*         y,
                                            int             incy);

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float
*alpha, const float *x, int incx,  float *y, int incy, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x,
int incx, float *y, int incy, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x,
int incx, double *y, int incy, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasSdot(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           const float*    y,
                                           int             incy,
                                           float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdot(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           const double*   y,
                                           int             incy,
                                           double*         result);

/*
HIPBLAS_EXPORT hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x,
int incx, const float *y, int incy, float *result, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x,
int incx, const double *y, int incy, double *result, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t
    hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx);

/* not implemented, requires complex support
hipblasStatus_t  hipblasCscal(hipblasHandle_t handle, int n, const hipComplex *alpha,  hipComplex
*x, int incx);

hipblasStatus_t  hipblasZscal(hipblasHandle_t handle, int n, const hipDoubleComplex *alpha,
hipDoubleComplex *x, int incx);
*/

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float
*alpha,  float *x, int incx, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double
*alpha,  double *x, int incx, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasSgemv(hipblasHandle_t    handle,
                                            hipblasOperation_t trans,
                                            int                m,
                                            int                n,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            const float*       x,
                                            int                incx,
                                            const float*       beta,
                                            float*             y,
                                            int                incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemv(hipblasHandle_t    handle,
                                            hipblasOperation_t trans,
                                            int                m,
                                            int                n,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            const double*      x,
                                            int                incx,
                                            const double*      beta,
                                            double*            y,
                                            int                incy);

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans,
int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int
batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasSger(hipblasHandle_t handle,
                                           int             m,
                                           int             n,
                                           const float*    alpha,
                                           const float*    x,
                                           int             incx,
                                           const float*    y,
                                           int             incy,
                                           float*          A,
                                           int             lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasDger(hipblasHandle_t handle,
                                           int             m,
                                           int             n,
                                           const double*   alpha,
                                           const double*   x,
                                           int             incx,
                                           const double*   y,
                                           int             incy,
                                           double*         A,
                                           int             lda);

/* not implemented
HIPBLAS_EXPORT hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float
*alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasStrsm(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const float*       alpha,
                                            float*             A,
                                            int                lda,
                                            float*             B,
                                            int                ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsm(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const double*      alpha,
                                            double*            A,
                                            int                lda,
                                            double*            B,
                                            int                ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasSgemm(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            const float*       B,
                                            int                ldb,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemm(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            const double*      B,
                                            int                ldb,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t transa,
                                                   hipblasOperation_t transb,
                                                   int                m,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float*       A[],
                                                   int                lda,
                                                   const float*       B[],
                                                   int                ldb,
                                                   const float*       beta,
                                                   float*             C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t transa,
                                                   hipblasOperation_t transb,
                                                   int                m,
                                                   int                n,
                                                   int                k,
                                                   const double*      alpha,
                                                   const double*      A[],
                                                   int                lda,
                                                   const double*      B[],
                                                   int                ldb,
                                                   const double*      beta,
                                                   double*            C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
                                             hipblasOperation_t trans_a,
                                             hipblasOperation_t trans_b,
                                             int                m,
                                             int                n,
                                             int                k,
                                             const void*        alpha,
                                             const void*        a,
                                             hipblasDatatype_t  a_type,
                                             int                lda,
                                             const void*        b,
                                             hipblasDatatype_t  b_type,
                                             int                ldb,
                                             const void*        beta,
                                             void*              c,
                                             hipblasDatatype_t  c_type,
                                             int                ldc,
                                             hipblasDatatype_t  compute_type,
                                             hipblasGemmAlgo_t  algo);

/* not implemented, requires complex support
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t
transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A, int
lda, const hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc);

hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t
transb,
                           int m, int n, int k,  const hipDoubleComplex *alpha, const
hipDoubleComplex *A, int lda, const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
hipDoubleComplex *C, int ldc);
*/

HIPBLAS_EXPORT hipblasStatus_t hipblasHgemm(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const hipblasHalf* alpha,
                                            const hipblasHalf* A,
                                            int                lda,
                                            const hipblasHalf* B,
                                            int                ldb,
                                            const hipblasHalf* beta,
                                            hipblasHalf*       C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t transa,
                                                          hipblasOperation_t transb,
                                                          int                m,
                                                          int                n,
                                                          int                k,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          long long          bsa,
                                                          const float*       B,
                                                          int                ldb,
                                                          long long          bsb,
                                                          const float*       beta,
                                                          float*             C,
                                                          int                ldc,
                                                          long long          bsc,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t transa,
                                                          hipblasOperation_t transb,
                                                          int                m,
                                                          int                n,
                                                          int                k,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          long long          bsa,
                                                          const double*      B,
                                                          int                ldb,
                                                          long long          bsb,
                                                          const double*      beta,
                                                          double*            C,
                                                          int                ldc,
                                                          long long          bsc,
                                                          int                batchCount);

#ifdef __cplusplus
}
#endif

#endif
