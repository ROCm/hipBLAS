/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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

struct hipblasBfloat16
{
    uint16_t data;
};

template <typename T>
struct hip_complex_number
{
    T x, y;

    template <typename U>
    hip_complex_number(U a, U b)
        : x(a)
        , y(b)
    {
    }
    template <typename U>
    hip_complex_number(U a)
        : x(a)
        , y(0)
    {
    }
    hip_complex_number()
        : x(0)
        , y(0)
    {
    }
};

typedef hip_complex_number<float>  hipblasComplex;
typedef hip_complex_number<double> hipblasDoubleComplex;

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
    HIPBLAS_R_16F = 150, /**< 16 bit floating point, real */
    HIPBLAS_R_32F = 151, /**< 32 bit floating point, real */
    HIPBLAS_R_64F = 152, /**< 64 bit floating point, real */
    HIPBLAS_C_16F = 153, /**< 16 bit floating point, complex */
    HIPBLAS_C_32F = 154, /**< 32 bit floating point, complex */
    HIPBLAS_C_64F = 155, /**< 64 bit floating point, complex */
    HIPBLAS_R_8I  = 160, /**<  8 bit signed integer, real */
    HIPBLAS_R_8U  = 161, /**<  8 bit unsigned integer, real */
    HIPBLAS_R_32I = 162, /**< 32 bit signed integer, real */
    HIPBLAS_R_32U = 163, /**< 32 bit unsigned integer, real */
    HIPBLAS_C_8I  = 164, /**<  8 bit signed integer, complex */
    HIPBLAS_C_8U  = 165, /**<  8 bit unsigned integer, complex */
    HIPBLAS_C_32I = 166, /**< 32 bit signed integer, complex */
    HIPBLAS_C_32U = 167, /**< 32 bit unsigned integer, complex */
    HIPBLAS_R_16B = 168, /**< 16 bit bfloat, real */
    HIPBLAS_C_16B = 169, /**< 16 bit bfloat, complex */
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

// amax
HIPBLAS_EXPORT hipblasStatus_t
    hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzamax(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

// amax_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batch_count,
                                                    int*                        result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    int                               batch_count,
                                                    int*                              result);

// amax_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const float*    x,
                                                           int             incx,
                                                           int             stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const double*   x,
                                                           int             incx,
                                                           int             stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batch_count,
                                                           int*                  result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           int                         batch_count,
                                                           int*                        result);

// amin
HIPBLAS_EXPORT hipblasStatus_t
    hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzamin(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

// amin_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasIsaminBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdaminBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batch_count,
                                                    int*                        result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    int                               batch_count,
                                                    int*                              result);

// amin_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const float*    x,
                                                           int             incx,
                                                           int             stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const double*   x,
                                                           int             incx,
                                                           int             stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batch_count,
                                                           int*                  result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           int                         batch_count,
                                                           int*                        result);

// asum
HIPBLAS_EXPORT hipblasStatus_t
    hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

// asum_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double* const x[],
                                                   int                 incx,
                                                   int                 batchCount,
                                                   double*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batchCount,
                                                    float*                      result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    int                               batchCount,
                                                    double*                           result);

// asum_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount,
                                                          float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount,
                                                          double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batchCount,
                                                           float*                result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           int                         batchCount,
                                                           double*                     result);

// axpy
HIPBLAS_EXPORT hipblasStatus_t hipblasHaxpy(hipblasHandle_t    handle,
                                            int                n,
                                            const hipblasHalf* alpha,
                                            const hipblasHalf* x,
                                            int                incx,
                                            hipblasHalf*       y,
                                            int                incy);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpy(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasComplex*       y,
                                            int                   incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpy(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// axpy_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHaxpyBatched(hipblasHandle_t          handle,
                                                   int                      n,
                                                   const hipblasHalf*       alpha,
                                                   const hipblasHalf* const x[],
                                                   int                      incx,
                                                   hipblasHalf* const       y[],
                                                   int                      incy,
                                                   int                      batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   int                incx,
                                                   float* const       y[],
                                                   int                incy,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   int                 incx,
                                                   double* const       y[],
                                                   int                 incy,
                                                   int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batch_count);

// axpy_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t    handle,
                                                          int                n,
                                                          const hipblasHalf* alpha,
                                                          const hipblasHalf* x,
                                                          int                incx,
                                                          int                stridex,
                                                          hipblasHalf*       y,
                                                          int                incy,
                                                          int                stridey,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    alpha,
                                                          const float*    x,
                                                          int             incx,
                                                          int             stridex,
                                                          float*          y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   alpha,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          double*         y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          int                         stridey,
                                                          int                         batch_count);

// copy
HIPBLAS_EXPORT hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasCcopy(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasZcopy(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// copy_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasScopyBatched(hipblasHandle_t    handle,
                                                   int                n,
                                                   const float* const x[],
                                                   int                incx,
                                                   float* const       y[],
                                                   int                incy,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double* const x[],
                                                   int                 incx,
                                                   double* const       y[],
                                                   int                 incy,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                                   int                               n,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batchCount);

// copy_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    x,
                                                          int             incx,
                                                          int             stridex,
                                                          float*          y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          double*         y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          int                         stridey,
                                                          int                         batchCount);

// dot
HIPBLAS_EXPORT hipblasStatus_t hipblasHdot(hipblasHandle_t    handle,
                                           int                n,
                                           const hipblasHalf* x,
                                           int                incx,
                                           const hipblasHalf* y,
                                           int                incy,
                                           hipblasHalf*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasBfdot(hipblasHandle_t        handle,
                                            int                    n,
                                            const hipblasBfloat16* x,
                                            int                    incx,
                                            const hipblasBfloat16* y,
                                            int                    incy,
                                            hipblasBfloat16*       result);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotc(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotu(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotc(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotu(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       result);

// dot_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHdotBatched(hipblasHandle_t          handle,
                                                  int                      n,
                                                  const hipblasHalf* const x[],
                                                  int                      incx,
                                                  const hipblasHalf* const y[],
                                                  int                      incy,
                                                  int                      batch_count,
                                                  hipblasHalf*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t              handle,
                                                   int                          n,
                                                   const hipblasBfloat16* const x[],
                                                   int                          incx,
                                                   const hipblasBfloat16* const y[],
                                                   int                          incy,
                                                   int                          batch_count,
                                                   hipblasBfloat16*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                                  int                n,
                                                  const float* const x[],
                                                  int                incx,
                                                  const float* const y[],
                                                  int                incy,
                                                  int                batch_count,
                                                  float*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdotBatched(hipblasHandle_t     handle,
                                                  int                 n,
                                                  const double* const x[],
                                                  int                 incx,
                                                  const double* const y[],
                                                  int                 incy,
                                                  int                 batch_count,
                                                  double*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   int                         batch_count,
                                                   hipblasComplex*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   int                         batch_count,
                                                   hipblasComplex*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                                   int                               n,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   int                               batch_count,
                                                   hipblasDoubleComplex*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                                   int                               n,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   int                               batch_count,
                                                   hipblasDoubleComplex*             result);

// dot_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t    handle,
                                                         int                n,
                                                         const hipblasHalf* x,
                                                         int                incx,
                                                         int                stridex,
                                                         const hipblasHalf* y,
                                                         int                incy,
                                                         int                stridey,
                                                         int                batch_count,
                                                         hipblasHalf*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                                          int                    n,
                                                          const hipblasBfloat16* x,
                                                          int                    incx,
                                                          int                    stridex,
                                                          const hipblasBfloat16* y,
                                                          int                    incy,
                                                          int                    stridey,
                                                          int                    batch_count,
                                                          hipblasBfloat16*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         const float*    x,
                                                         int             incx,
                                                         int             stridex,
                                                         const float*    y,
                                                         int             incy,
                                                         int             stridey,
                                                         int             batch_count,
                                                         float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         const double*   x,
                                                         int             incx,
                                                         int             stridex,
                                                         const double*   y,
                                                         int             incy,
                                                         int             stridey,
                                                         int             batch_count,
                                                         double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count,
                                                          hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batch_count,
                                                          hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          int                         stridey,
                                                          int                         batch_count,
                                                          hipblasDoubleComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          int                         stridey,
                                                          int                         batch_count,
                                                          hipblasDoubleComplex*       result);

// snrm2
HIPBLAS_EXPORT hipblasStatus_t
    hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

// nrm2_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                                   int                 n,
                                                   const double* const x[],
                                                   int                 incx,
                                                   int                 batchCount,
                                                   double*             result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const hipblasComplex* const x[],
                                                    int                         incx,
                                                    int                         batchCount,
                                                    float*                      result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const x[],
                                                    int                               incx,
                                                    int                               batchCount,
                                                    double*                           result);

// nrm2_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount,
                                                          float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount,
                                                          double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batchCount,
                                                           float*                result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           int                         stridex,
                                                           int                         batchCount,
                                                           double*                     result);

// rot
HIPBLAS_EXPORT hipblasStatus_t hipblasSrot(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           float*          y,
                                           int             incy,
                                           const float*    c,
                                           const float*    s);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrot(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           double*         y,
                                           int             incy,
                                           const double*   c,
                                           const double*   s);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrot(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           const float*          c,
                                           const hipblasComplex* s);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,
                                            int             n,
                                            hipblasComplex* x,
                                            int             incx,
                                            hipblasComplex* y,
                                            int             incy,
                                            const float*    c,
                                            const float*    s);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrot(hipblasHandle_t             handle,
                                           int                         n,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           const double*               c,
                                           const hipblasDoubleComplex* s);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdrot(hipblasHandle_t       handle,
                                            int                   n,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasDoubleComplex* y,
                                            int                   incy,
                                            const double*         c,
                                            const double*         s);

// rot_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotBatched(hipblasHandle_t handle,
                                                  int             n,
                                                  float* const    x[],
                                                  int             incx,
                                                  float* const    y[],
                                                  int             incy,
                                                  const float*    c,
                                                  const float*    s,
                                                  int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotBatched(hipblasHandle_t handle,
                                                  int             n,
                                                  double* const   x[],
                                                  int             incx,
                                                  double* const   y[],
                                                  int             incy,
                                                  const double*   c,
                                                  const double*   s,
                                                  int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotBatched(hipblasHandle_t       handle,
                                                  int                   n,
                                                  hipblasComplex* const x[],
                                                  int                   incx,
                                                  hipblasComplex* const y[],
                                                  int                   incy,
                                                  const float*          c,
                                                  const hipblasComplex* s,
                                                  int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t       handle,
                                                   int                   n,
                                                   hipblasComplex* const x[],
                                                   int                   incx,
                                                   hipblasComplex* const y[],
                                                   int                   incy,
                                                   const float*          c,
                                                   const float*          s,
                                                   int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotBatched(hipblasHandle_t             handle,
                                                  int                         n,
                                                  hipblasDoubleComplex* const x[],
                                                  int                         incx,
                                                  hipblasDoubleComplex* const y[],
                                                  int                         incy,
                                                  const double*               c,
                                                  const hipblasDoubleComplex* s,
                                                  int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   hipblasDoubleComplex* const x[],
                                                   int                         incx,
                                                   hipblasDoubleComplex* const y[],
                                                   int                         incy,
                                                   const double*               c,
                                                   const double*               s,
                                                   int                         batchCount);

// rot_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         float*          x,
                                                         int             incx,
                                                         int             stridex,
                                                         float*          y,
                                                         int             incy,
                                                         int             stridey,
                                                         const float*    c,
                                                         const float*    s,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         double*         x,
                                                         int             incx,
                                                         int             stridex,
                                                         double*         y,
                                                         int             incy,
                                                         int             stridey,
                                                         const double*   c,
                                                         const double*   s,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                                         int                   n,
                                                         hipblasComplex*       x,
                                                         int                   incx,
                                                         int                   stridex,
                                                         hipblasComplex*       y,
                                                         int                   incy,
                                                         int                   stridey,
                                                         const float*          c,
                                                         const hipblasComplex* s,
                                                         int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          hipblasComplex* x,
                                                          int             incx,
                                                          int             stridex,
                                                          hipblasComplex* y,
                                                          int             incy,
                                                          int             stridey,
                                                          const float*    c,
                                                          const float*    s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                                         int                         n,
                                                         hipblasDoubleComplex*       x,
                                                         int                         incx,
                                                         int                         stridex,
                                                         hipblasDoubleComplex*       y,
                                                         int                         incy,
                                                         int                         stridey,
                                                         const double*               c,
                                                         const hipblasDoubleComplex* s,
                                                         int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          hipblasDoubleComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasDoubleComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          const double*         c,
                                                          const double*         s,
                                                          int                   batchCount);

// rotg
HIPBLAS_EXPORT hipblasStatus_t
    hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotg(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotg(hipblasHandle_t       handle,
                                            hipblasDoubleComplex* a,
                                            hipblasDoubleComplex* b,
                                            double*               c,
                                            hipblasDoubleComplex* s);

// rotg_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                                   float* const    a[],
                                                   float* const    b[],
                                                   float* const    c[],
                                                   float* const    s[],
                                                   int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                                   double* const   a[],
                                                   double* const   b[],
                                                   double* const   c[],
                                                   double* const   s[],
                                                   int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                                   hipblasComplex* const a[],
                                                   hipblasComplex* const b[],
                                                   float* const          c[],
                                                   hipblasComplex* const s[],
                                                   int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                                   hipblasDoubleComplex* const a[],
                                                   hipblasDoubleComplex* const b[],
                                                   double* const               c[],
                                                   hipblasDoubleComplex* const s[],
                                                   int                         batchCount);

// rotg_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle,
                                                          float*          a,
                                                          int             stride_a,
                                                          float*          b,
                                                          int             stride_b,
                                                          float*          c,
                                                          int             stride_c,
                                                          float*          s,
                                                          int             stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                                          double*         a,
                                                          int             stride_a,
                                                          double*         b,
                                                          int             stride_b,
                                                          double*         c,
                                                          int             stride_c,
                                                          double*         s,
                                                          int             stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                                          hipblasComplex* a,
                                                          int             stride_a,
                                                          hipblasComplex* b,
                                                          int             stride_b,
                                                          float*          c,
                                                          int             stride_c,
                                                          hipblasComplex* s,
                                                          int             stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                                          hipblasDoubleComplex* a,
                                                          int                   stride_a,
                                                          hipblasDoubleComplex* b,
                                                          int                   stride_b,
                                                          double*               c,
                                                          int                   stride_c,
                                                          hipblasDoubleComplex* s,
                                                          int                   stride_s,
                                                          int                   batchCount);

// rotm
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotm(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotm(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);

// rotm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmBatched(hipblasHandle_t    handle,
                                                   int                n,
                                                   float* const       x[],
                                                   int                incx,
                                                   float* const       y[],
                                                   int                incy,
                                                   const float* const param[],
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t     handle,
                                                   int                 n,
                                                   double* const       x[],
                                                   int                 incx,
                                                   double* const       y[],
                                                   int                 incy,
                                                   const double* const param[],
                                                   int                 batchCount);

// rotm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          float*          x,
                                                          int             incx,
                                                          int             stridex,
                                                          float*          y,
                                                          int             incy,
                                                          int             stridey,
                                                          const float*    param,
                                                          int             stride_param,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          double*         x,
                                                          int             incx,
                                                          int             stridex,
                                                          double*         y,
                                                          int             incy,
                                                          int             stridey,
                                                          const double*   param,
                                                          int             stride_param,
                                                          int             batchCount);

// rotmg
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);

// rotmg_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmgBatched(hipblasHandle_t    handle,
                                                    float* const       d1[],
                                                    float* const       d2[],
                                                    float* const       x1[],
                                                    const float* const y1[],
                                                    float* const       param[],
                                                    int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                                    double* const       d1[],
                                                    double* const       d2[],
                                                    double* const       x1[],
                                                    const double* const y1[],
                                                    double* const       param[],
                                                    int                 batchCount);

// rotmg_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle,
                                                           float*          d1,
                                                           int             stride_d1,
                                                           float*          d2,
                                                           int             stride_d2,
                                                           float*          x1,
                                                           int             stride_x1,
                                                           const float*    y1,
                                                           int             stride_y1,
                                                           float*          param,
                                                           int             stride_param,
                                                           int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                                           double*         d1,
                                                           int             stride_d1,
                                                           double*         d2,
                                                           int             stride_d2,
                                                           double*         x1,
                                                           int             stride_x1,
                                                           const double*   y1,
                                                           int             stride_y1,
                                                           double*         param,
                                                           int             stride_param,
                                                           int             batchCount);

// scal
HIPBLAS_EXPORT hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCscal(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasZscal(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdscal(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);

// scal_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDscalBatched(hipblasHandle_t handle,
                                                   int             n,
                                                   const double*   alpha,
                                                   double* const   x[],
                                                   int             incx,
                                                   int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                                   int                   n,
                                                   const hipblasComplex* alpha,
                                                   hipblasComplex* const x[],
                                                   int                   incx,
                                                   int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                                   int                         n,
                                                   const hipblasDoubleComplex* alpha,
                                                   hipblasDoubleComplex* const x[],
                                                   int                         incx,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                                    int                   n,
                                                    const float*          alpha,
                                                    hipblasComplex* const x[],
                                                    int                   incx,
                                                    int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                                    int                         n,
                                                    const double*               alpha,
                                                    hipblasDoubleComplex* const x[],
                                                    int                         incx,
                                                    int                         batchCount);

// scal_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    alpha,
                                                          float*          x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   alpha,
                                                          double*         x,
                                                          int             incx,
                                                          int             stridex,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const float*    alpha,
                                                           hipblasComplex* x,
                                                           int             incx,
                                                           int             stridex,
                                                           int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const double*         alpha,
                                                           hipblasDoubleComplex* x,
                                                           int                   incx,
                                                           int                   stridex,
                                                           int                   batchCount);

// swap
HIPBLAS_EXPORT hipblasStatus_t
    hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t
    hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasCswap(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasZswap(hipblasHandle_t       handle,
                                            int                   n,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasDoubleComplex* y,
                                            int                   incy);

// swap_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSswapBatched(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDswapBatched(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle,
                                                   int             n,
                                                   hipblasComplex* x[],
                                                   int             incx,
                                                   hipblasComplex* y[],
                                                   int             incy,
                                                   int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZswapBatched(hipblasHandle_t       handle,
                                                   int                   n,
                                                   hipblasDoubleComplex* x[],
                                                   int                   incx,
                                                   hipblasDoubleComplex* y[],
                                                   int                   incy,
                                                   int                   batchCount);

// swap_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          float*          x,
                                                          int             incx,
                                                          int             stridex,
                                                          float*          y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          double*         x,
                                                          int             incx,
                                                          int             stridex,
                                                          double*         y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          hipblasComplex* x,
                                                          int             incx,
                                                          int             stridex,
                                                          hipblasComplex* y,
                                                          int             incy,
                                                          int             stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          hipblasDoubleComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          hipblasDoubleComplex* y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batchCount);

// gemv
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemv(hipblasHandle_t       handle,
                                            hipblasOperation_t    trans,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemv(hipblasHandle_t             handle,
                                            hipblasOperation_t          trans,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// gemv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t trans,
                                                   int                m,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float*       beta,
                                                   float* const       y[],
                                                   int                incy,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t     handle,
                                                   hipblasOperation_t  trans,
                                                   int                 m,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double*       beta,
                                                   double* const       y[],
                                                   int                 incy,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t             handle,
                                                   hipblasOperation_t          trans,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t                   handle,
                                                   hipblasOperation_t                trans,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batchCount);

// gemv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t trans,
                                                          int                m,
                                                          int                n,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          int                strideA,
                                                          const float*       x,
                                                          int                incx,
                                                          int                stridex,
                                                          const float*       beta,
                                                          float*             y,
                                                          int                incy,
                                                          int                stridey,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t trans,
                                                          int                m,
                                                          int                n,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          int                strideA,
                                                          const double*      x,
                                                          int                incx,
                                                          int                stridex,
                                                          const double*      beta,
                                                          double*            y,
                                                          int                stridey,
                                                          int                incy,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasOperation_t    trans,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          int                   strideA,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          int                   stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          int                   stridey,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasOperation_t          trans,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          int                         strideA,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          int                         stridex,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         stridey,
                                                          int                         incy,
                                                          int                         batchCount);

// ger
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

// ger_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgerBatched(hipblasHandle_t    handle,
                                                  int                m,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float* const x[],
                                                  int                incx,
                                                  const float* const y[],
                                                  int                incy,
                                                  float* const       A[],
                                                  int                lda,
                                                  int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgerBatched(hipblasHandle_t     handle,
                                                  int                 m,
                                                  int                 n,
                                                  const double*       alpha,
                                                  const double* const x[],
                                                  int                 incx,
                                                  const double* const y[],
                                                  int                 incy,
                                                  double* const       A[],
                                                  int                 lda,
                                                  int                 batchCount);

// ger_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle,
                                                         int             m,
                                                         int             n,
                                                         const float*    alpha,
                                                         const float*    x,
                                                         int             incx,
                                                         int             stridex,
                                                         const float*    y,
                                                         int             incy,
                                                         int             stridey,
                                                         float*          A,
                                                         int             lda,
                                                         int             strideA,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
                                                         int             m,
                                                         int             n,
                                                         const double*   alpha,
                                                         const double*   x,
                                                         int             incx,
                                                         int             stridex,
                                                         const double*   y,
                                                         int             incy,
                                                         int             stridey,
                                                         double*         A,
                                                         int             lda,
                                                         int             strideA,
                                                         int             batchCount);

// syr
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           float*            A,
                                           int               lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           double*           A,
                                           int               lda);

// syr_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrBatched(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float* const x[],
                                                  int                incx,
                                                  float* const       A[],
                                                  int                lda,
                                                  int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrBatched(hipblasHandle_t     handle,
                                                  hipblasFillMode_t   uplo,
                                                  int                 n,
                                                  const double*       alpha,
                                                  const double* const x[],
                                                  int                 incx,
                                                  double* const       A[],
                                                  int                 lda,
                                                  int                 batchCount);

// syr_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const float*      alpha,
                                                         const float*      x,
                                                         int               incx,
                                                         int               stridex,
                                                         float*            A,
                                                         int               lda,
                                                         int               stridey,
                                                         int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const double*     alpha,
                                                         const double*     x,
                                                         int               incx,
                                                         int               stridex,
                                                         double*           A,
                                                         int               lda,
                                                         int               stridey,
                                                         int               batchCount);

// trsv
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const float*       A,
                                            int                lda,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const double*      A,
                                            int                lda,
                                            double*            x,
                                            int                incx);

// trsv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   const float* const A[],
                                                   int                lda,
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   const double* const A[],
                                                   int                 lda,
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batch_count);

// trsv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const float*       A,
                                                          int                lda,
                                                          int                strideA,
                                                          float*             x,
                                                          int                incx,
                                                          int                stridex,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const double*      A,
                                                          int                lda,
                                                          int                strideA,
                                                          double*            x,
                                                          int                incx,
                                                          int                stridex,
                                                          int                batch_count);

// trmm
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            float*             B,
                                            int                ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmm(hipblasHandle_t    handle,
                                            hipblasSideMode_t  side,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                n,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            double*            B,
                                            int                ldb);

// trsm
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

// trsm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                n,
                                                   const float*       alpha,
                                                   float* const       A[],
                                                   int                lda,
                                                   float*             B[],
                                                   int                ldb,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                n,
                                                   const double*      alpha,
                                                   double* const      A[],
                                                   int                lda,
                                                   double*            B[],
                                                   int                ldb,
                                                   int                batch_count);

// trsm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasSideMode_t  side,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                n,
                                                          const float*       alpha,
                                                          float*             A,
                                                          int                lda,
                                                          int                strideA,
                                                          float*             B,
                                                          int                ldb,
                                                          int                strideB,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasSideMode_t  side,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                n,
                                                          const double*      alpha,
                                                          double*            A,
                                                          int                lda,
                                                          int                strideA,
                                                          double*            B,
                                                          int                ldb,
                                                          int                strideB,
                                                          int                batch_count);

// gemm
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemm(hipblasHandle_t       handle,
                                            hipblasOperation_t    transa,
                                            hipblasOperation_t    transb,
                                            int                   m,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemm(hipblasHandle_t             handle,
                                            hipblasOperation_t          transa,
                                            hipblasOperation_t          transb,
                                            int                         m,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// gemm batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t          handle,
                                                   hipblasOperation_t       transa,
                                                   hipblasOperation_t       transb,
                                                   int                      m,
                                                   int                      n,
                                                   int                      k,
                                                   const hipblasHalf*       alpha,
                                                   const hipblasHalf* const A[],
                                                   int                      lda,
                                                   const hipblasHalf* const B[],
                                                   int                      ldb,
                                                   const hipblasHalf*       beta,
                                                   hipblasHalf* const       C[],
                                                   int                      ldc,
                                                   int                      batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t transa,
                                                   hipblasOperation_t transb,
                                                   int                m,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const B[],
                                                   int                ldb,
                                                   const float*       beta,
                                                   float* const       C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t     handle,
                                                   hipblasOperation_t  transa,
                                                   hipblasOperation_t  transb,
                                                   int                 m,
                                                   int                 n,
                                                   int                 k,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const B[],
                                                   int                 ldb,
                                                   const double*       beta,
                                                   double* const       C[],
                                                   int                 ldc,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t             handle,
                                                   hipblasOperation_t          transa,
                                                   hipblasOperation_t          transb,
                                                   int                         m,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const B[],
                                                   int                         ldb,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t                   handle,
                                                   hipblasOperation_t                transa,
                                                   hipblasOperation_t                transb,
                                                   int                               m,
                                                   int                               n,
                                                   int                               k,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const B[],
                                                   int                               ldb,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batchCount);

// gemm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t transa,
                                                          hipblasOperation_t transb,
                                                          int                m,
                                                          int                n,
                                                          int                k,
                                                          const hipblasHalf* alpha,
                                                          const hipblasHalf* A,
                                                          int                lda,
                                                          long long          bsa,
                                                          const hipblasHalf* B,
                                                          int                ldb,
                                                          long long          bsb,
                                                          const hipblasHalf* beta,
                                                          hipblasHalf*       C,
                                                          int                ldc,
                                                          long long          bsc,
                                                          int                batchCount);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasOperation_t    transa,
                                                          hipblasOperation_t    transb,
                                                          int                   m,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          long long             bsa,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          long long             bsb,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          long long             bsc,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasOperation_t          transa,
                                                          hipblasOperation_t          transb,
                                                          int                         m,
                                                          int                         n,
                                                          int                         k,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          long long                   bsa,
                                                          const hipblasDoubleComplex* B,
                                                          int                         ldb,
                                                          long long                   bsb,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          long long                   bsc,
                                                          int                         batchCount);

// gemmex
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

#ifdef __cplusplus
}
#endif

#endif
