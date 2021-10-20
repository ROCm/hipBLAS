/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
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

#include "hipblas-export.h"
#include "hipblas-version.h"
#include <hip/hip_runtime_api.h>
#include <stdint.h>

/* Workaround clang bug:

   https://bugs.llvm.org/show_bug.cgi?id=35863

   This macro expands to static if clang is used; otherwise it expands empty.
   It is intended to be used in variable template specializations, where clang
   requires static in order for the specializations to have internal linkage,
   while technically, storage class specifiers besides thread_local are not
   allowed in template specializations, and static in the primary template
   definition should imply internal linkage for all specializations.

   If clang shows an error for improperly using a storage class specifier in
   a specialization, then HIPBLAS_CLANG_STATIC should be redefined as empty,
   and perhaps removed entirely, if the above bug has been fixed.
*/
#if __clang__
#define HIPBLAS_CLANG_STATIC static
#else
#define HIPBLAS_CLANG_STATIC
#endif

#ifndef HIPBLAS_DEPRECATED_MSG
#define HIPBLAS_DEPRECATED_MSG(MSG) __attribute__((deprecated(#MSG)))
#endif

typedef void* hipblasHandle_t;

typedef uint16_t hipblasHalf;

typedef int8_t hipblasInt8;

typedef int64_t hipblasStride;

typedef struct hipblasBfloat16
{
    uint16_t data;
} hipblasBfloat16;

#if defined(ROCM_MATHLIBS_API_USE_HIP_COMPLEX)
// Using hip complex types

#include <hip/hip_complex.h>

/*! \brief hip type to represent a complex number with single precision real and imaginary parts. */
typedef hipFloatComplex hipblasComplex;

/*! \brief hip type to represent a complex number with double precision real and imaginary parts. */
typedef hipDoubleComplex hipblasDoubleComplex;

#else
// using internal complex class for API

typedef struct hipblasComplex
{
#ifndef __cplusplus

    float x, y;

#else

private:
    float x, y;

public:
#if __cplusplus >= 201103L
    hipblasComplex() = default;
#else
    hipblasComplex() {}
#endif

    hipblasComplex(float r, float i = 0)
        : x(r)
        , y(i)
    {
    }

    float real() const
    {
        return x;
    }
    float imag() const
    {
        return y;
    }
    void real(float r)
    {
        x = r;
    }
    void imag(float i)
    {
        y = i;
    }

#endif
} hipblasComplex;

typedef struct hipblasDoubleComplex
{
#ifndef __cplusplus

    double x, y;

#else

private:
    double x, y;

public:

#if __cplusplus >= 201103L
    hipblasDoubleComplex() = default;
#else
    hipblasDoubleComplex() {}
#endif

    hipblasDoubleComplex(double r, double i = 0)
        : x(r)
        , y(i)
    {
    }
    double real() const
    {
        return x;
    }
    double imag() const
    {
        return y;
    }
    void real(double r)
    {
        x = r;
    }
    void imag(double i)
    {
        y = i;
    }

#endif
} hipblasDoubleComplex;

// this isn't needed right now
// typedef struct hipblasInt8Complex
// {
// #ifndef __cplusplus

//     hipblasInt8 x, y;

// #else

// private:
//     hipblasInt8 x, y;

// public:
// #if __cplusplus >= 201103L
//     hipblasInt8Complex() = default;
// #else
//     hipblasInt8Complex() {}
// #endif

//     hipblasInt8Complex(hipblasInt8 r, hipblasInt8 i = 0)
//         : x(r)
//         , y(i)
//     {
//     }

//     hipblasInt8 real() const
//     {
//         return x;
//     }
//     hipblasInt8 imag() const
//     {
//         return y;
//     }
//     void real(hipblasInt8 r)
//     {
//         x = r;
//     }
//     void imag(hipblasInt8 i)
//     {
//         y = i;
//     }

// #endif
// } hipblasInt8Complex;

#if __cplusplus >= 201103L
static_assert(std::is_trivial<hipblasComplex>{},
              "hipblasComplex is not a trivial type, and thus is incompatible with C.");
static_assert(std::is_trivial<hipblasDoubleComplex>{},
              "hipblasDoubleComplex is not a trivial type, and thus is incompatible with C.");
#endif

#endif // using internal complex class for API

#if __cplusplus >= 201103L
#include <type_traits>
static_assert(std::is_standard_layout<hipblasComplex>{},
              "hipblasComplex is not a standard layout type, and thus is incompatible with C.");
static_assert(
    std::is_standard_layout<hipblasDoubleComplex>{},
    "hipblasDoubleComplex is not a standard layout type, and thus is incompatible with C.");
static_assert(sizeof(hipblasComplex) == sizeof(float) * 2
                  && sizeof(hipblasDoubleComplex) == sizeof(double) * 2
                  && sizeof(hipblasDoubleComplex) == sizeof(hipblasComplex) * 2,
              "Sizes of hipblasComplex or hipblasDoubleComplex are inconsistent");
#endif

typedef enum
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
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9, // hipBLAS handle is null pointer
    HIPBLAS_STATUS_INVALID_ENUM      = 10, // unsupported enum value was passed to function
    HIPBLAS_STATUS_UNKNOWN           = 11, // back-end returned an unsupported status code
} hipblasStatus_t;

// set the values of enum constants to be the same as those used in cblas
typedef enum
{
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113,
} hipblasOperation_t;

typedef enum
{
    HIPBLAS_POINTER_MODE_HOST,
    HIPBLAS_POINTER_MODE_DEVICE,
} hipblasPointerMode_t;

typedef enum
{
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL  = 123,
} hipblasFillMode_t;

typedef enum
{
    HIPBLAS_DIAG_NON_UNIT = 131,
    HIPBLAS_DIAG_UNIT     = 132,
} hipblasDiagType_t;

typedef enum
{
    HIPBLAS_SIDE_LEFT  = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH  = 143,
} hipblasSideMode_t;

typedef enum
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
} hipblasDatatype_t;

typedef enum
{
    HIPBLAS_GEMM_DEFAULT = 160,
} hipblasGemmAlgo_t;

typedef enum
{
    HIPBLAS_ATOMICS_NOT_ALLOWED = 0,
    HIPBLAS_ATOMICS_ALLOWED     = 1,
} hipblasAtomicsMode_t;

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

HIPBLAS_EXPORT hipblasStatus_t hipblasSetVectorAsync(
    int n, int elem_size, const void* x, int incx, void* y, int incy, hipStream_t stream);

HIPBLAS_EXPORT hipblasStatus_t hipblasGetVectorAsync(
    int n, int elem_size, const void* x, int incx, void* y, int incy, hipStream_t stream);

HIPBLAS_EXPORT hipblasStatus_t hipblasSetMatrixAsync(int         rows,
                                                     int         cols,
                                                     int         elem_size,
                                                     const void* A,
                                                     int         lda,
                                                     void*       B,
                                                     int         ldb,
                                                     hipStream_t stream);

HIPBLAS_EXPORT hipblasStatus_t hipblasGetMatrixAsync(int         rows,
                                                     int         cols,
                                                     int         elem_size,
                                                     const void* A,
                                                     int         lda,
                                                     void*       B,
                                                     int         ldb,
                                                     hipStream_t stream);

HIPBLAS_EXPORT hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t      handle,
                                                     hipblasAtomicsMode_t atomics_mode);

HIPBLAS_EXPORT hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t       handle,
                                                     hipblasAtomicsMode_t* atomics_mode);

/*! \brief BLAS Level 1 API

    \details
    amax finds the first index of the element of maximum magnitude of a vector x.
    vector

    @param[in]
    handle    [hipblasHandle_t]
              handle to the hipblas library context queue.
    @param[in]
    n         [int]
              the number of elements in x.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [int]
              specifies the increment for the elements of y.
    @param[inout]
    result
              device pointer or host pointer to store the amax index.
              return is 0.0 if n, incx<=0.
    ********************************************************************/
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
                                                           hipblasStride   stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const double*   x,
                                                           int             incx,
                                                           hipblasStride   stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           hipblasStride         stridex,
                                                           int                   batch_count,
                                                           int*                  result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           hipblasStride               stridex,
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
                                                           hipblasStride   stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const double*   x,
                                                           int             incx,
                                                           hipblasStride   stridex,
                                                           int             batch_count,
                                                           int*            result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           hipblasStride         stridex,
                                                           int                   batch_count,
                                                           int*                  result);

HIPBLAS_EXPORT hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           hipblasStride               stridex,
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
                                                          hipblasStride   stridex,
                                                          int             batchCount,
                                                          float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          int             batchCount,
                                                          double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           hipblasStride         stridex,
                                                           int                   batchCount,
                                                           float*                result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           hipblasStride               stridex,
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
                                                          hipblasStride      stridex,
                                                          hipblasHalf*       y,
                                                          int                incy,
                                                          hipblasStride      stridey,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const float*    alpha,
                                                          const float*    x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          float*          y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   alpha,
                                                          const double*   x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          double*         y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
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
                                                          hipblasStride   stridex,
                                                          float*          y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          double*         y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
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
                                                         hipblasStride      stridex,
                                                         const hipblasHalf* y,
                                                         int                incy,
                                                         hipblasStride      stridey,
                                                         int                batch_count,
                                                         hipblasHalf*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                                          int                    n,
                                                          const hipblasBfloat16* x,
                                                          int                    incx,
                                                          hipblasStride          stridex,
                                                          const hipblasBfloat16* y,
                                                          int                    incy,
                                                          hipblasStride          stridey,
                                                          int                    batch_count,
                                                          hipblasBfloat16*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         const float*    x,
                                                         int             incx,
                                                         hipblasStride   stridex,
                                                         const float*    y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         int             batch_count,
                                                         float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         const double*   x,
                                                         int             incx,
                                                         hipblasStride   stridex,
                                                         const double*   y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         int             batch_count,
                                                         double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batch_count,
                                                          hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batch_count,
                                                          hipblasComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          int                         batch_count,
                                                          hipblasDoubleComplex*       result);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
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
                                                          hipblasStride   stridex,
                                                          int             batchCount,
                                                          float*          result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          int             batchCount,
                                                          double*         result);

HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const hipblasComplex* x,
                                                           int                   incx,
                                                           hipblasStride         stridex,
                                                           int                   batchCount,
                                                           float*                result);

HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                                           int                         n,
                                                           const hipblasDoubleComplex* x,
                                                           int                         incx,
                                                           hipblasStride               stridex,
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
                                                         hipblasStride   stridex,
                                                         float*          y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         const float*    c,
                                                         const float*    s,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                                         int             n,
                                                         double*         x,
                                                         int             incx,
                                                         hipblasStride   stridex,
                                                         double*         y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         const double*   c,
                                                         const double*   s,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                                         int                   n,
                                                         hipblasComplex*       x,
                                                         int                   incx,
                                                         hipblasStride         stridex,
                                                         hipblasComplex*       y,
                                                         int                   incy,
                                                         hipblasStride         stridey,
                                                         const float*          c,
                                                         const hipblasComplex* s,
                                                         int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          hipblasComplex* x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          hipblasComplex* y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          const float*    c,
                                                          const float*    s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                                         int                         n,
                                                         hipblasDoubleComplex*       x,
                                                         int                         incx,
                                                         hipblasStride               stridex,
                                                         hipblasDoubleComplex*       y,
                                                         int                         incy,
                                                         hipblasStride               stridey,
                                                         const double*               c,
                                                         const hipblasDoubleComplex* s,
                                                         int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          hipblasDoubleComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          hipblasDoubleComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
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
                                                          hipblasStride   stride_a,
                                                          float*          b,
                                                          hipblasStride   stride_b,
                                                          float*          c,
                                                          hipblasStride   stride_c,
                                                          float*          s,
                                                          hipblasStride   stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                                          double*         a,
                                                          hipblasStride   stride_a,
                                                          double*         b,
                                                          hipblasStride   stride_b,
                                                          double*         c,
                                                          hipblasStride   stride_c,
                                                          double*         s,
                                                          hipblasStride   stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                                          hipblasComplex* a,
                                                          hipblasStride   stride_a,
                                                          hipblasComplex* b,
                                                          hipblasStride   stride_b,
                                                          float*          c,
                                                          hipblasStride   stride_c,
                                                          hipblasComplex* s,
                                                          hipblasStride   stride_s,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                                          hipblasDoubleComplex* a,
                                                          hipblasStride         stride_a,
                                                          hipblasDoubleComplex* b,
                                                          hipblasStride         stride_b,
                                                          double*               c,
                                                          hipblasStride         stride_c,
                                                          hipblasDoubleComplex* s,
                                                          hipblasStride         stride_s,
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
                                                          hipblasStride   stridex,
                                                          float*          y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          const float*    param,
                                                          hipblasStride   stride_param,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          double*         x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          double*         y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          const double*   param,
                                                          hipblasStride   stride_param,
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
                                                           hipblasStride   stride_d1,
                                                           float*          d2,
                                                           hipblasStride   stride_d2,
                                                           float*          x1,
                                                           hipblasStride   stride_x1,
                                                           const float*    y1,
                                                           hipblasStride   stride_y1,
                                                           float*          param,
                                                           hipblasStride   stride_param,
                                                           int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                                           double*         d1,
                                                           hipblasStride   stride_d1,
                                                           double*         d2,
                                                           hipblasStride   stride_d2,
                                                           double*         x1,
                                                           hipblasStride   stride_x1,
                                                           const double*   y1,
                                                           hipblasStride   stride_y1,
                                                           double*         param,
                                                           hipblasStride   stride_param,
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
                                                          hipblasStride   stridex,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          const double*   alpha,
                                                          double*         x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                                           int             n,
                                                           const float*    alpha,
                                                           hipblasComplex* x,
                                                           int             incx,
                                                           hipblasStride   stridex,
                                                           int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                                           int                   n,
                                                           const double*         alpha,
                                                           hipblasDoubleComplex* x,
                                                           int                   incx,
                                                           hipblasStride         stridex,
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
                                                          hipblasStride   stridex,
                                                          float*          y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          double*         x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          double*         y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                                          int             n,
                                                          hipblasComplex* x,
                                                          int             incx,
                                                          hipblasStride   stridex,
                                                          hipblasComplex* y,
                                                          int             incy,
                                                          hipblasStride   stridey,
                                                          int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                                          int                   n,
                                                          hipblasDoubleComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          hipblasDoubleComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

// ================================
// ========== LEVEL 2 =============
// ================================

// gbmv
HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmv(hipblasHandle_t    handle,
                                            hipblasOperation_t trans,
                                            int                m,
                                            int                n,
                                            int                kl,
                                            int                ku,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            const float*       x,
                                            int                incx,
                                            const float*       beta,
                                            float*             y,
                                            int                incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmv(hipblasHandle_t    handle,
                                            hipblasOperation_t trans,
                                            int                m,
                                            int                n,
                                            int                kl,
                                            int                ku,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            const double*      x,
                                            int                incx,
                                            const double*      beta,
                                            double*            y,
                                            int                incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmv(hipblasHandle_t       handle,
                                            hipblasOperation_t    trans,
                                            int                   m,
                                            int                   n,
                                            int                   kl,
                                            int                   ku,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

/*! \brief BLAS Level 2 API

    \details
    xGBMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of A
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device pointer storing banded matrix A.
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be >= (kl + ku + 1)
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmv(hipblasHandle_t             handle,
                                            hipblasOperation_t          trans,
                                            int                         m,
                                            int                         n,
                                            int                         kl,
                                            int                         ku,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// gbmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmvBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t trans,
                                                   int                m,
                                                   int                n,
                                                   int                kl,
                                                   int                ku,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float*       beta,
                                                   float* const       y[],
                                                   int                incy,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmvBatched(hipblasHandle_t     handle,
                                                   hipblasOperation_t  trans,
                                                   int                 m,
                                                   int                 n,
                                                   int                 kl,
                                                   int                 ku,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double*       beta,
                                                   double* const       y[],
                                                   int                 incy,
                                                   int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmvBatched(hipblasHandle_t             handle,
                                                   hipblasOperation_t          trans,
                                                   int                         m,
                                                   int                         n,
                                                   int                         kl,
                                                   int                         ku,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGBMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of each matrix A_i
    @param[in]
    n         [rocblas_int]
              number of columns of each matrix A_i
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of each A_i
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of each A_i
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device array of device pointers storing each banded matrix A_i.
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be >= (kl + ku + 1)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmvBatched(hipblasHandle_t                   handle,
                                                   hipblasOperation_t                trans,
                                                   int                               m,
                                                   int                               n,
                                                   int                               kl,
                                                   int                               ku,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batch_count);

// gbmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t trans,
                                                          int                m,
                                                          int                n,
                                                          int                kl,
                                                          int                ku,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          const float*       x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          const float*       beta,
                                                          float*             y,
                                                          int                incy,
                                                          hipblasStride      stride_y,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t trans,
                                                          int                m,
                                                          int                n,
                                                          int                kl,
                                                          int                ku,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          const double*      x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          const double*      beta,
                                                          double*            y,
                                                          int                incy,
                                                          hipblasStride      stride_y,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasOperation_t    trans,
                                                          int                   m,
                                                          int                   n,
                                                          int                   kl,
                                                          int                   ku,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         stride_a,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stride_x,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stride_y,
                                                          int                   batch_count);

/*! \brief BLAS Level 2 API

    \details
    xGBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n banded matrix with kl sub-diagonals and ku super-diagonals,
    for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    kl        [rocblas_int]
              number of sub-diagonals of A
    @param[in]
    ku        [rocblas_int]
              number of super-diagonals of A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
        A     device pointer to first banded matrix (A_1).
              Leading (kl + ku + 1) by n part of the matrix contains the coefficients
              of the banded matrix. The leading diagonal resides in row (ku + 1) with
              the first super-diagonal above on the RHS of row ku. The first sub-diagonal
              resides below on the LHS of row ku + 2. This propogates up and down across
              sub/super-diagonals.
                Ex: (m = n = 7; ku = 2, kl = 2)
                1 2 3 0 0 0 0             0 0 3 3 3 3 3
                4 1 2 3 0 0 0             0 2 2 2 2 2 2
                5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
                0 5 4 1 2 3 0             4 4 4 4 4 4 0
                0 0 5 4 1 2 0             5 5 5 5 5 0 0
                0 0 0 5 4 1 2             0 0 0 0 0 0 0
                0 0 0 0 5 4 1             0 0 0 0 0 0 0
              Note that the empty elements which don't correspond to data will not
              be referenced.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be >= (kl + ku + 1)
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         device pointer to first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1)
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer to first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (x_i+1)
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasOperation_t          trans,
                                                          int                         m,
                                                          int                         n,
                                                          int                         kl,
                                                          int                         ku,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               stride_a,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stride_x,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stride_y,
                                                          int                         batch_count);

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

/*! \brief BLAS Level 2 API

    \details
    xGEMV performs one of the matrix-vector operations

        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not
    @param[in]
    m         [rocblas_int]
              number of rows of matrix A
    @param[in]
    n         [rocblas_int]
              number of columns of matrix A
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
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

/*! \brief BLAS Level 2 API

    \details
    xGEMV_BATCHED performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    trans       [rocblas_operation]
                indicates whether matrices A_i are tranposed (conjugated) or not
    @param[in]
    m           [rocblas_int]
                number of rows of each matrix A_i
    @param[in]
    n           [rocblas_int]
                number of columns of each matrix A_i
    @param[in]
    alpha       device pointer or host pointer to scalar alpha.
    @param[in]
    A           device array of device pointers storing each matrix A_i.
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of each matrix A_i.
    @param[in]
    x           device array of device pointers storing each vector x_i.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of each vector x_i.
    @param[in]
    beta        device pointer or host pointer to scalar beta.
    @param[inout]
    y           device array of device pointers storing each vector y_i.
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of each vector y_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
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
                                                          hipblasStride      strideA,
                                                          const float*       x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          const float*       beta,
                                                          float*             y,
                                                          int                incy,
                                                          hipblasStride      stridey,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t trans,
                                                          int                m,
                                                          int                n,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          const double*      x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          const double*      beta,
                                                          double*            y,
                                                          int                incy,
                                                          hipblasStride      stridey,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasOperation_t    trans,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xGEMV_STRIDED_BATCHED performs a batch of matrix-vector operations

        y_i := alpha*A_i*x_i    + beta*y_i,   or
        y_i := alpha*A_i**T*x_i + beta*y_i,   or
        y_i := alpha*A_i**H*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle      [rocblas_handle]
                handle to the rocblas library context queue.
    @param[in]
    transA      [rocblas_operation]
                indicates whether matrices A_i are tranposed (conjugated) or not
    @param[in]
    m           [rocblas_int]
                number of rows of matrices A_i
    @param[in]
    n           [rocblas_int]
                number of columns of matrices A_i
    @param[in]
    alpha       device pointer or host pointer to scalar alpha.
    @param[in]
    A           device pointer to the first matrix (A_1) in the batch.
    @param[in]
    lda         [rocblas_int]
                specifies the leading dimension of matrices A_i.
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x           device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx        [rocblas_int]
                specifies the increment for the elements of vectors x_i.
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stride_x is of appropriate size. When trans equals rocblas_operation_none
                this typically means stride_x >= n * incx, otherwise stride_x >= m * incx.
    @param[in]
    beta        device pointer or host pointer to scalar beta.
    @param[inout]
    y           device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy        [rocblas_int]
                specifies the increment for the elements of vectors y_i.
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stride_y, however the user should
                take care to ensure that stride_y is of appropriate size. When trans equals rocblas_operation_none
                this typically means stride_y >= m * incy, otherwise stride_y >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasOperation_t          trans,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeru(hipblasHandle_t       handle,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       A,
                                            int                   lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgerc(hipblasHandle_t       handle,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       A,
                                            int                   lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeru(hipblasHandle_t             handle,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       A,
                                            int                         lda);

/*! \brief BLAS Level 2 API

    \details
    xGER,xGERU,xGERC performs the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for xGERC

    where alpha is a scalar, x and y are vectors, and A is an
    m by n matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of the matrix A.
    @param[in]
    n         [rocblas_int]
              the number of columns of the matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/

HIPBLAS_EXPORT hipblasStatus_t hipblasZgerc(hipblasHandle_t             handle,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       A,
                                            int                         lda);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeruBatched(hipblasHandle_t             handle,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       A[],
                                                   int                         lda,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgercBatched(hipblasHandle_t             handle,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       A[],
                                                   int                         lda,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeruBatched(hipblasHandle_t                   handle,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       A[],
                                                   int                               lda,
                                                   int                               batchCount);

/*! \brief BLAS Level 2 API

    \details
    xGER,xGERU,xGERC_BATCHED performs a batch of the matrix-vector operations

        A := A + alpha*x*y**T , OR
        A := A + alpha*x*y**H for xGERC

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of each matrix A_i.
    @param[in]
    n         [rocblas_int]
              the number of columns of eaceh matrix A_i.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

HIPBLAS_EXPORT hipblasStatus_t hipblasZgercBatched(hipblasHandle_t                   handle,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       A[],
                                                   int                               lda,
                                                   int                               batchCount);

// ger_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle,
                                                         int             m,
                                                         int             n,
                                                         const float*    alpha,
                                                         const float*    x,
                                                         int             incx,
                                                         hipblasStride   stridex,
                                                         const float*    y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         float*          A,
                                                         int             lda,
                                                         hipblasStride   strideA,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
                                                         int             m,
                                                         int             n,
                                                         const double*   alpha,
                                                         const double*   x,
                                                         int             incx,
                                                         hipblasStride   stridex,
                                                         const double*   y,
                                                         int             incy,
                                                         hipblasStride   stridey,
                                                         double*         A,
                                                         int             lda,
                                                         hipblasStride   strideA,
                                                         int             batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t       handle,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t       handle,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t             handle,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xGERC,xGERU,xGERC_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**T, OR
        A_i := A_i + alpha*x_i*y_i**H  for xGERC

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha is a scalar, x_i and y_i are vectors and A_i is an
    m by n matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    m         [rocblas_int]
              the number of rows of each matrix A_i.
    @param[in]
    n         [rocblas_int]
              the number of columns of each matrix A_i.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector (x_1) in the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increments for the elements of each vector x_i.
    @param[in]
    stridex   [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
              There are no restrictions placed on stride_x, however the user should
              take care to ensure that stride_x is of appropriate size, for a typical
              case this means stride_x >= m * incx.
    @param[inout]
    y         device pointer to the first vector (y_1) in the batch.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i.
    @param[in]
    stridey   [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
              There are no restrictions placed on stride_y, however the user should
              take care to ensure that stride_y is of appropriate size, for a typical
              case this means stride_y >= n * incy.
    @param[inout]
    A         device pointer to the first matrix (A_1) in the batch.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/

HIPBLAS_EXPORT hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t             handle,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          int                         batchCount);

// hbmv
HIPBLAS_EXPORT hipblasStatus_t hipblasChbmv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

/*! \brief BLAS Level 2 API

    \details
    xHBMV performs the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian band matrix, with k super-diagonals.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of A is being supplied.
              rocblas_fill_lower: The lower triangular part of A is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of the matrix A. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A. Of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of A must contain the upper
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of A will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of A must contain the lower
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of A will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a Hermitian matrix, the imaginary part of the main diagonal
              of A will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. must be >= k + 1
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// hbmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChbmvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHBMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of each A_i is being supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of each matrix A_i. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix_i A of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of each A_i must contain the upper
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of each A_i will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of each A_i must contain the lower
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of each A_i will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a Hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   int                               k,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batchCount);

// hbmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: The upper triangular part of each A_i is being supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is being supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    k         [rocblas_int]
              the number of super-diagonals of each matrix A_i. Must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array pointing to the first matrix A_1. Each A_i is of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The leading (k + 1) by n part of each A_i must contain the upper
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (k + 1), the first super-diagonal on the RHS
                of row k, etc.
                The top left k by x triangle of each A_i will not be referenced.
                    Ex (upper, lda = n = 4, k = 1):
                    A                             Represented matrix
                    (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
                    (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
                    (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)

              if uplo == rocblas_fill_lower:
                The leading (k + 1) by n part of each A_i must contain the lower
                triangular band part of the Hermitian matrix, with the leading
                diagonal in row (1), the first sub-diagonal on the LHS of
                row 2, etc.
                The bottom right k by k triangle of each A_i will not be referenced.
                    Ex (lower, lda = 2, n = 4, k = 1):
                    A                               Represented matrix
                    (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
                    (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
                                                    (0, 0) (6, 8) (3, 0) (7,-7)
                                                    (0, 0) (0, 0) (7, 7) (4, 0)

              As a Hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         device array pointing to the first vector y_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1)
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array pointing to the first vector y_1.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          int                         k,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          int                         batchCount);

// hemv
HIPBLAS_EXPORT hipblasStatus_t hipblasChemv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

/*! \brief BLAS Level 2 API

    \details
    xHEMV performs one of the matrix-vector operations

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the Hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the Hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device pointer storing matrix A. Of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of A must contain
                the upper triangular part of a Hermitian matrix. The lower
                triangular part of A will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of A must contain
                the lower triangular part of a Hermitian matrix. The upper
                triangular part of A will not be referenced.
              As a Hermitian matrix, the imaginary part of the main diagonal
              of A will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. must be >= max(1, n)
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhemv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         da,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// hemv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChemvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHEMV_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the Hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the Hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of each A_i must contain
                the upper triangular part of a Hermitian matrix. The lower
                triangular part of each A_i will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of each A_i must contain
                the lower triangular part of a Hermitian matrix. The upper
                triangular part of each A_i will not be referenced.
              As a Hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhemvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batch_count);

// hemv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         stride_a,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stride_x,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stride_y,
                                                          int                   batch_count);

/*! \brief BLAS Level 2 API

    \details
    xHEMV_STRIDED_BATCHED performs one of the matrix-vector operations

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the Hermitian matrix A is supplied.
              rocblas_fill_lower: the lower triangular part of the Hermitian matrix A is supplied.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    A         device array of device pointers storing each matrix A_i of dimension (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular part of each A_i must contain
                the upper triangular part of a Hermitian matrix. The lower
                triangular part of each A_i will not be referenced.
              if uplo == rocblas_fill_lower:
                The lower triangular part of each A_i must contain
                the lower triangular part of a Hermitian matrix. The upper
                triangular part of each A_i will not be referenced.
              As a Hermitian matrix, the imaginary part of the main diagonal
              of each A_i will not be referenced and is assumed to be == 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. must be >= max(1, n)
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               stride_a,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stride_x,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stride_y,
                                                          int                         batch_count);

// her
HIPBLAS_EXPORT hipblasStatus_t hipblasCher(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const float*          alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasComplex*       A,
                                           int                   lda);

/*! \brief BLAS Level 2 API

    \details
    xHER performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n Hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in A.
              rocblas_fill_lower: The lower triangular part of A is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    A         device pointer storing the specified triangular portion of
              the Hermitian matrix A. Of size (lda * n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the Hermitian matrix A is supplied. The lower
                triangluar portion will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the Hermitian matrix A is supplied. The upper
                triangular portion will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be at least max(1, n).
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZher(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const double*               alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasDoubleComplex*       A,
                                           int                         lda);

// her_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherBatched(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const float*                alpha,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  hipblasComplex* const       A[],
                                                  int                         lda,
                                                  int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHER_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in A.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular portion
                of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular portion
                of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(1, n).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZherBatched(hipblasHandle_t                   handle,
                                                  hipblasFillMode_t                 uplo,
                                                  int                               n,
                                                  const double*                     alpha,
                                                  const hipblasDoubleComplex* const x[],
                                                  int                               incx,
                                                  hipblasDoubleComplex* const       A[],
                                                  int                               lda,
                                                  int                               batchCount);

// her_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t       handle,
                                                         hipblasFillMode_t     uplo,
                                                         int                   n,
                                                         const float*          alpha,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         hipblasStride         stridex,
                                                         hipblasComplex*       A,
                                                         int                   lda,
                                                         hipblasStride         strideA,
                                                         int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHER_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n Hermitian matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in A.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in A.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const double*               alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasStride               stridex,
                                                         hipblasDoubleComplex*       A,
                                                         int                         lda,
                                                         hipblasStride               strideA,
                                                         int                         batchCount);

// her2
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       A,
                                            int                   lda);

/*! \brief BLAS Level 2 API

    \details
    xHER2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n Hermitian matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied.
              rocblas_fill_lower: The lower triangular part of A is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing the specified triangular portion of
              the Hermitian matrix A. Of size (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
                portion of A will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
                portion of A will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. Must be at least max(lda, 1).
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZher2(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       A,
                                            int                         lda);

// her2_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2Batched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       A[],
                                                   int                         lda,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHER2_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n Hermitian matrix for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    A         device array of device pointers storing the specified triangular portion of
              each Hermitian matrix A_i of size (lda, n).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(lda, 1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZher2Batched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       A[],
                                                   int                               lda,
                                                   int                               batchCount);

// her2_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHER2_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n Hermitian matrix for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              specifies the stride between the beginning of one vector (x_i) and the next (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              specifies the stride between the beginning of one vector (y_i) and the next (y_i+1).
    @param[inout]
    A         device pointer pointing to the first matrix (A_1). Stores the specified triangular portion of
              each Hermitian matrix A_i.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
                portion of each A_i will not be touched.
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
                portion of each A_i will not be touched.
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. Must be at least max(lda, 1).
    @param[in]
    stride_A  [rocblas_stride]
              specifies the stride between the beginning of one matrix (A_i) and the next (A_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          int                         batchCount);

// hpmv
HIPBLAS_EXPORT hipblasStatus_t hipblasChpmv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* AP,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

/*! \brief BLAS Level 2 API

    \details
    xHPMV performs the matrix-vector operation

        y := alpha*A*x + beta*y

    where alpha and beta are scalars, x and y are n element vectors and A is an
    n by n Hermitian matrix, supplied in packed form (see description below).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of the Hermitian matrix A is supplied in AP.
              rocblas_fill_lower: the lower triangular part of the Hermitian matrix A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of the matrix A, must be >= 0.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP        device pointer storing the packed version of the specified triangular portion of
              the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* AP,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// hpmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChpmvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const AP[],
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       y[],
                                                   int                         incy,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPMV_BATCHED performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
              rocblas_fill_lower: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP      device pointer of device pointers storing the packed version of the specified triangular
            portion of each Hermitian matrix A_i. Each A_i is of at least size ((n * (n + 1)) / 2).
            if uplo == rocblas_fill_upper:
            The upper triangular portion of each Hermitian matrix A_i is supplied.
            The matrix is compacted so that each AP_i contains the triangular portion column-by-column
            so that:
            AP(0) = A(0,0)
            AP(1) = A(0,1)
            AP(2) = A(1,1), etc.
                Ex: (rocblas_fill_upper; n = 3)
                    (1, 0) (2, 1) (3, 2)
                    (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                    (3,-2) (5, 1) (6, 0)
        if uplo == rocblas_fill_lower:
            The lower triangular portion of each Hermitian matrix A_i is supplied.
            The matrix is compacted so that each AP_i contains the triangular portion column-by-column
            so that:
            AP(0) = A(0,0)
            AP(1) = A(1,0)
            AP(2) = A(2,1), etc.
                Ex: (rocblas_fill_lower; n = 3)
                    (1, 0) (2, 1) (3, 2)
                    (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                    (3,-2) (5, 1) (6, 0)
        Note that the imaginary part of the diagonal elements are not accessed and are assumed
        to be 0.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const AP[],
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       y[],
                                                   int                               incy,
                                                   int                               batchCount);

// hpmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* AP,
                                                          hipblasStride         strideAP,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPMV_STRIDED_BATCHED performs the matrix-vector operation

        y_i := alpha*A_i*x_i + beta*y_i

    where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
    n by n Hermitian matrix, supplied in packed form (see description below),
    for each batch in i = [1, batch_count].

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
              rocblas_fill_lower: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the order of each matrix A_i.
    @param[in]
    alpha     device pointer or host pointer to scalar alpha.
    @param[in]
    AP        device pointer pointing to the beginning of the first matrix (AP_1). Stores the packed
              version of the specified triangular portion of each Hermitian matrix AP_i of size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that each AP_i contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that each AP_i contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (3, 2)
                        (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
                        (3,-2) (5, 1) (6, 0)
        Note that the imaginary part of the diagonal elements are not accessed and are assumed
        to be 0.
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one matrix (AP_i) and the next one (AP_i+1).
    @param[in]
    x         device array pointing to the beginning of the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    beta      device pointer or host pointer to scalar beta.
    @param[inout]
    y         device array pointing to the beginning of the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* AP,
                                                          hipblasStride               strideAP,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          int                         batchCount);

// hpr
HIPBLAS_EXPORT hipblasStatus_t hipblasChpr(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const float*          alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasComplex*       AP);

/*! \brief BLAS Level 2 API

    \details
    xHPR performs the matrix-vector operations

        A := A + alpha*x*x**H

    where alpha is a real scalar, x is a vector, and A is an
    n by n Hermitian matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const double*               alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasDoubleComplex*       AP);

// hpr_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChprBatched(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const float*                alpha,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  hipblasComplex* const       AP[],
                                                  int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPR_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhprBatched(hipblasHandle_t                   handle,
                                                  hipblasFillMode_t                 uplo,
                                                  int                               n,
                                                  const double*                     alpha,
                                                  const hipblasDoubleComplex* const x[],
                                                  int                               incx,
                                                  hipblasDoubleComplex* const       AP[],
                                                  int                               batchCount);

// hpr_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t       handle,
                                                         hipblasFillMode_t     uplo,
                                                         int                   n,
                                                         const float*          alpha,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         hipblasStride         stridex,
                                                         hipblasComplex*       AP,
                                                         hipblasStride         strideAP,
                                                         int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPR_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**H

    where alpha is a real scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const double*               alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasStride               stridex,
                                                         hipblasDoubleComplex*       AP,
                                                         hipblasStride               strideAP,
                                                         int                         batchCount);

// hpr2
HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       AP);

/*! \brief BLAS Level 2 API

    \details
    xHPR2 performs the matrix-vector operations

        A := A + alpha*x*y**H + conj(alpha)*y*x**H

    where alpha is a complex scalar, x and y are vectors, and A is an
    n by n Hermitian matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the Hermitian matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       AP);

// hpr2_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2Batched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       AP[],
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPR2_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       AP[],
                                                   int                               batchCount);

// hpr2_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          hipblasComplex*       AP,
                                                          hipblasStride         strideAP,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xHPR2_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H

    where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each Hermitian matrix A_i. Points to the first matrix (A_1).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each Hermitian matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 3)
                        (1, 0) (2, 1) (4,9)
                        (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
                        (4,-9) (5,-3) (6,0)
            Note that the imaginary part of the diagonal elements are not accessed and are assumed
            to be 0.
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          hipblasDoubleComplex*       AP,
                                                          hipblasStride               strideAP,
                                                          int                         batchCount);

// sbmv
HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            int               k,
                                            const float*      alpha,
                                            const float*      A,
                                            int               lda,
                                            const float*      x,
                                            int               incx,
                                            const float*      beta,
                                            float*            y,
                                            int               incy);

/*! \brief BLAS Level 2 API

    \details
    xSBMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of matrix A
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            int               k,
                                            const double*     alpha,
                                            const double*     A,
                                            int               lda,
                                            const double*     x,
                                            int               incx,
                                            const double*     beta,
                                            double*           y,
                                            int               incy);

// sbmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float*       beta,
                                                   float*             y[],
                                                   int                incy,
                                                   int                batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSBMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   int                 n,
                                                   int                 k,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double*       beta,
                                                   double*             y[],
                                                   int                 incy,
                                                   int                 batchCount);

// sbmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          int               k,
                                                          const float*      alpha,
                                                          const float*      A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const float*      beta,
                                                          float*            y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSBMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric banded matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n symmetric banded matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    k         [rocblas_int]
              specifies the number of sub- and super-diagonals
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stridex, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stridey, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          int               k,
                                                          const double*     alpha,
                                                          const double*     A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const double*     beta,
                                                          double*           y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

// spmv
HIPBLAS_EXPORT hipblasStatus_t hipblasSspmv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const float*      alpha,
                                            const float*      AP,
                                            const float*      x,
                                            int               incx,
                                            const float*      beta,
                                            float*            y,
                                            int               incy);

/*! \brief BLAS Level 2 API

    \details
    xSPMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspmv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const double*     alpha,
                                            const double*     AP,
                                            const double*     x,
                                            int               incx,
                                            const double*     beta,
                                            double*           y,
                                            int               incy);

// TODO: Complex
// spmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSspmvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const AP[],
                                                   const float* const x[],
                                                   int                incx,
                                                   const float*       beta,
                                                   float*             y[],
                                                   int                incy,
                                                   int                batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const AP[],
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double*       beta,
                                                   double*             y[],
                                                   int                 incy,
                                                   int                 batchCount);

// spmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const float*      alpha,
                                                          const float*      AP,
                                                          hipblasStride     strideAP,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const float*      beta,
                                                          float*            y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A should contain an upper or lower triangular n by n packed symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stridex, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stridey, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const double*     alpha,
                                                          const double*     AP,
                                                          hipblasStride     strideAP,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const double*     beta,
                                                          double*           y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

// spr
HIPBLAS_EXPORT hipblasStatus_t hipblasSspr(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           float*            AP);

HIPBLAS_EXPORT hipblasStatus_t hipblasDspr(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           double*           AP);

HIPBLAS_EXPORT hipblasStatus_t hipblasCspr(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasComplex*       AP);

/*! \brief BLAS Level 2 API

    \details
    xSPR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasDoubleComplex*       AP);

// spr_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsprBatched(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  int                n,
                                                  const float*       alpha,
                                                  const float* const x[],
                                                  int                incx,
                                                  float* const       AP[],
                                                  int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsprBatched(hipblasHandle_t     handle,
                                                  hipblasFillMode_t   uplo,
                                                  int                 n,
                                                  const double*       alpha,
                                                  const double* const x[],
                                                  int                 incx,
                                                  double* const       AP[],
                                                  int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsprBatched(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasComplex*       alpha,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  hipblasComplex* const       AP[],
                                                  int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPR_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsprBatched(hipblasHandle_t                   handle,
                                                  hipblasFillMode_t                 uplo,
                                                  int                               n,
                                                  const hipblasDoubleComplex*       alpha,
                                                  const hipblasDoubleComplex* const x[],
                                                  int                               incx,
                                                  hipblasDoubleComplex* const       AP[],
                                                  int                               batchCount);

// spr_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const float*      alpha,
                                                         const float*      x,
                                                         int               incx,
                                                         hipblasStride     stridex,
                                                         float*            AP,
                                                         hipblasStride     strideAP,
                                                         int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const double*     alpha,
                                                         const double*     x,
                                                         int               incx,
                                                         hipblasStride     stridex,
                                                         double*           AP,
                                                         hipblasStride     strideAP,
                                                         int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t       handle,
                                                         hipblasFillMode_t     uplo,
                                                         int                   n,
                                                         const hipblasComplex* alpha,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         hipblasStride         stridex,
                                                         hipblasComplex*       AP,
                                                         hipblasStride         strideAP,
                                                         int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPR_STRIDED_BATCHED performs the matrix-vector operations

        A_i := A_i + alpha*x_i*x_i**T

    where alpha is a scalar, x_i is a vector, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(2) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const hipblasDoubleComplex* alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasStride               stridex,
                                                         hipblasDoubleComplex*       AP,
                                                         hipblasStride               strideAP,
                                                         int                         batchCount);

// spr2
HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const float*      alpha,
                                            const float*      x,
                                            int               incx,
                                            const float*      y,
                                            int               incy,
                                            float*            AP);

/*! \brief BLAS Level 2 API

    \details
    xSPR2 performs the matrix-vector operation

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix, supplied in packed form.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of A is supplied in AP.
              rocblas_fill_lower: The lower triangular part of A is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
              if uplo == rocblas_fill_upper:
                The upper triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of the symmetric matrix A is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const double*     alpha,
                                            const double*     x,
                                            int               incx,
                                            const double*     y,
                                            int               incy,
                                            double*           AP);

// spr2_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2Batched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float* const y[],
                                                   int                incy,
                                                   float* const       AP[],
                                                   int                batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPR2_BATCHED performs the matrix-vector operation

        A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T

    where alpha is a scalar, x_i and y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    AP        device array of device pointers storing the packed version of the specified triangular portion of
              each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batch_count.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2Batched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double* const y[],
                                                   int                 incy,
                                                   double* const       AP[],
                                                   int                 batchCount);

// spr2_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const float*      alpha,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const float*      y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          float*            AP,
                                                          hipblasStride     strideAP,
                                                          int               batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSPR2_STRIDED_BATCHED performs the matrix-vector operation

        A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T

    where alpha is a scalar, x_i amd y_i are vectors, and A_i is an
    n by n symmetric matrix, supplied in packed form, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              rocblas_fill_upper: The upper triangular part of each A_i is supplied in AP.
              rocblas_fill_lower: The lower triangular part of each A_i is supplied in AP.
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A_i, must be at least 0.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer pointing to the first vector (x_1).
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one vector (x_i) and the next one (x_i+1).
    @param[in]
    y         device pointer pointing to the first vector (y_1).
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stride_y  [rocblas_stride]
              stride from the start of one vector (y_i) and the next one (y_i+1).
    @param[inout]
    AP        device pointer storing the packed version of the specified triangular portion of
              each symmetric matrix A_i. Points to the first A_1.
              if uplo == rocblas_fill_upper:
                The upper triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(0,1)
                AP(2) = A(1,1), etc.
                    Ex: (rocblas_fill_upper; n = 4)
                        1 2 4 7
                        2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        4 5 6 9
                        7 8 9 0
            if uplo == rocblas_fill_lower:
                The lower triangular portion of each symmetric matrix A_i is supplied.
                The matrix is compacted so that AP contains the triangular portion column-by-column
                so that:
                AP(0) = A(0,0)
                AP(1) = A(1,0)
                AP(n) = A(2,1), etc.
                    Ex: (rocblas_fill_lower; n = 4)
                        1 2 3 4
                        2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
                        3 6 8 9
                        4 7 9 0
    @param[in]
    stride_A    [rocblas_stride]
                stride from the start of one (A_i) and the next (A_i+1)
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.
    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const double*     alpha,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const double*     y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          double*           AP,
                                                          hipblasStride     strideAP,
                                                          int               batchCount);

// symv
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const float*      alpha,
                                            const float*      A,
                                            int               lda,
                                            const float*      x,
                                            int               incx,
                                            const float*      beta,
                                            float*            y,
                                            int               incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymv(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const double*     alpha,
                                            const double*     A,
                                            int               lda,
                                            const double*     x,
                                            int               incx,
                                            const double*     beta,
                                            double*           y,
                                            int               incy);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       y,
                                            int                   incy);

/*! \brief BLAS Level 2 API

    \details
    xSYMV performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A should contain an upper or lower triangular n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      rocblas_fill
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
    @param[in]
    alpha
              specifies the scalar alpha
    @param[in]
    A         pointer storing matrix A on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A
    @param[in]
    x         pointer storing vector x on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x
    @param[in]
    beta      specifies the scalar beta
    @param[out]
    y         pointer storing vector y on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsymv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       y,
                                            int                         incy);

// symv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float*       beta,
                                                   float*             y[],
                                                   int                incy,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double*       beta,
                                                   double*             y[],
                                                   int                 incy,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex*             y[],
                                                   int                         incy,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYMV_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         device array of device pointers storing each matrix A_i
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    x         device array of device pointers storing each vector x_i
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         device array of device pointers storing each vector y_i
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsymvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex*             y[],
                                                   int                               incy,
                                                   int                               batchCount);

// symv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const float*      alpha,
                                                          const float*      A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const float*      beta,
                                                          float*            y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const double*     alpha,
                                                          const double*     A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const double*     beta,
                                                          double*           y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYMV_strided_batched performs the matrix-vector operation:

        y_i := alpha*A_i*x_i + beta*y_i,

    where (A_i, x_i, y_i) is the i-th instance of the batch.
    alpha and beta are scalars, x_i and y_i are vectors and A_i is an
    n by n symmetric matrix, for i = 1, ..., batch_count.
    A a should contain an upper or lower triangular symmetric matrix
    and the opposing triangular part of A is not referenced

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              number of rows and columns of each matrix A_i
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha
    @param[in]
    A         Device pointer to the first matrix A_1 on the GPU
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each matrix A_i
    @param[in]
    strideA     [rocblas_stride]
                stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    x         Device pointer to the first vector x_1 on the GPU
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each vector x_i
    @param[in]
    stridex     [rocblas_stride]
                stride from the start of one vector (x_i) and the next one (x_i+1).
                There are no restrictions placed on stride_x, however the user should
                take care to ensure that stridex is of appropriate size.
                This typically means stridex >= n * incx. stridex should be non zero.
    @param[in]
    beta      device pointer or host pointer to scalar beta
    @param[out]
    y         Device pointer to the first vector y_1 on the GPU
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each vector y_i
    @param[in]
    stridey     [rocblas_stride]
                stride from the start of one vector (y_i) and the next one (y_i+1).
                There are no restrictions placed on stride_y, however the user should
                take care to ensure that stridey is of appropriate size.
                This typically means stridey >= n * incy. stridey should be non zero.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          int                         batchCount);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasComplex*       A,
                                           int                   lda);

/*! \brief BLAS Level 2 API

    \details
    xSYR performs the matrix-vector operations

        A := A + alpha*x*x**T

    where alpha is a scalar, x is a vector, and A is an
    n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced

    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasDoubleComplex*       A,
                                           int                         lda);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrBatched(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int                         n,
                                                  const hipblasComplex*       alpha,
                                                  const hipblasComplex* const x[],
                                                  int                         incx,
                                                  hipblasComplex* const       A[],
                                                  int                         lda,
                                                  int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYR_batched performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, x is an array of vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrBatched(hipblasHandle_t                   handle,
                                                  hipblasFillMode_t                 uplo,
                                                  int                               n,
                                                  const hipblasDoubleComplex*       alpha,
                                                  const hipblasDoubleComplex* const x[],
                                                  int                               incx,
                                                  hipblasDoubleComplex* const       A[],
                                                  int                               lda,
                                                  int                               batchCount);

// syr_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const float*      alpha,
                                                         const float*      x,
                                                         int               incx,
                                                         hipblasStride     stridex,
                                                         float*            A,
                                                         int               lda,
                                                         hipblasStride     stridey,
                                                         int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t   handle,
                                                         hipblasFillMode_t uplo,
                                                         int               n,
                                                         const double*     alpha,
                                                         const double*     x,
                                                         int               incx,
                                                         hipblasStride     stridex,
                                                         double*           A,
                                                         int               lda,
                                                         hipblasStride     stridey,
                                                         int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t       handle,
                                                         hipblasFillMode_t     uplo,
                                                         int                   n,
                                                         const hipblasComplex* alpha,
                                                         const hipblasComplex* x,
                                                         int                   incx,
                                                         hipblasStride         stridex,
                                                         hipblasComplex*       A,
                                                         int                   lda,
                                                         hipblasStride         stridey,
                                                         int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYR_strided_batched performs the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*x[i]**T

    where alpha is a scalar, vectors, and A is an array of
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between vectors (x_i) and (x_i+1).
    @param[inout]
    A         device pointer to the first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA   [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t             handle,
                                                         hipblasFillMode_t           uplo,
                                                         int                         n,
                                                         const hipblasDoubleComplex* alpha,
                                                         const hipblasDoubleComplex* x,
                                                         int                         incx,
                                                         hipblasStride               stridex,
                                                         hipblasDoubleComplex*       A,
                                                         int                         lda,
                                                         hipblasStride               stridey,
                                                         int                         batchCount);

// syr2
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const float*      alpha,
                                            const float*      x,
                                            int               incx,
                                            const float*      y,
                                            int               incy,
                                            float*            A,
                                            int               lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int               n,
                                            const double*     alpha,
                                            const double*     x,
                                            int               incx,
                                            const double*     y,
                                            int               incy,
                                            double*           A,
                                            int               lda);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            const hipblasComplex* y,
                                            int                   incy,
                                            hipblasComplex*       A,
                                            int                   lda);

/*! \brief BLAS Level 2 API

    \details
    xSYR2 performs the matrix-vector operations

        A := A + alpha*x*y**T + alpha*y*x**T

    where alpha is a scalar, x and y are vectors, and A is an
    n by n symmetric matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced

    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.
    @param[in]
    y         device pointer storing vector y.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of y.
    @param[inout]
    A         device pointer storing matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            const hipblasDoubleComplex* y,
                                            int                         incy,
                                            hipblasDoubleComplex*       A,
                                            int                         lda);

// syr2_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2Batched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const x[],
                                                   int                incx,
                                                   const float* const y[],
                                                   int                incy,
                                                   float* const       A[],
                                                   int                lda,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2Batched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const x[],
                                                   int                 incx,
                                                   const double* const y[],
                                                   int                 incy,
                                                   double* const       A[],
                                                   int                 lda,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2Batched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   const hipblasComplex* const y[],
                                                   int                         incy,
                                                   hipblasComplex* const       A[],
                                                   int                         lda,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYR2_BATCHED performs a batch of matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrix, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device array of device pointers storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    y         device array of device pointers storing each vector y_i.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[inout]
    A         device array of device pointers storing each matrix A_i.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2Batched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   const hipblasDoubleComplex* const y[],
                                                   int                               incy,
                                                   hipblasDoubleComplex* const       A[],
                                                   int                               lda,
                                                   int                               batchCount);

// syr2_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const float*      alpha,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const float*      y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          float*            A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t   handle,
                                                          hipblasFillMode_t uplo,
                                                          int               n,
                                                          const double*     alpha,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const double*     y,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          double*           A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          const hipblasComplex* y,
                                                          int                   incy,
                                                          hipblasStride         stridey,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    xSYR2_STRIDED_BATCHED the matrix-vector operations

        A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T

    where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
    n by n symmetric matrices, for i = 1 , ... , batch_count

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              specifies whether the upper 'rocblas_fill_upper' or lower 'rocblas_fill_lower'
              if rocblas_fill_upper, the lower part of A is not referenced
              if rocblas_fill_lower, the upper part of A is not referenced
    @param[in]
    n         [rocblas_int]
              the number of rows and columns of each matrix A.
    @param[in]
    alpha
              device pointer or host pointer to scalar alpha.
    @param[in]
    x         device pointer to the first vector x_1.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stridex   [rocblas_stride]
              specifies the pointer increment between vectors (x_i) and (x_i+1).
    @param[in]
    y         device pointer to the first vector y_1.
    @param[in]
    incy      [rocblas_int]
              specifies the increment for the elements of each y_i.
    @param[in]
    stridey   [rocblas_stride]
              specifies the pointer increment between vectors (y_i) and (y_i+1).
    @param[inout]
    A         device pointer to the first matrix A_1.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
    @param[in]
    strideA   [rocblas_stride]
              stride from the start of one matrix (A_i) and the next one (A_i+1)
    @param[in]
    batch_count [rocblas_int]
              number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          const hipblasDoubleComplex* y,
                                                          int                         incy,
                                                          hipblasStride               stridey,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          int                         batchCount);

// tbmv
HIPBLAS_EXPORT hipblasStatus_t hipblasStbmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                k,
                                            const float*       A,
                                            int                lda,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            int                k,
                                            const double*      A,
                                            int                lda,
                                            double*            x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            int                   k,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    xTBMV performs one of the matrix-vector operations

        x := A*x      or
        x := A**T*x   or
        x := A**H*x,

    x is a vectors and A is a banded m by m matrix (see description below).

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: A is an upper banded triangular matrix.
              rocblas_fill_lower: A is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether matrix A is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of A is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of A's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by A.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of the matrix A.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of the matrix A.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device pointer storing banded triangular matrix A.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A. lda must satisfy lda > k.
    @param[inout]
    x         device pointer storing vector x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            int                         k,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

// tbmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStbmvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                k,
                                                   const float* const A[],
                                                   int                lda,
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   int                 k,
                                                   const double* const A[],
                                                   int                 lda,
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   int                         k,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count);

/*! \brief BLAS Level 2 API

    \details
    xTBMV_BATCHED performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: each A_i is an upper banded triangular matrix.
              rocblas_fill_lower: each A_i is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether each matrix A_i is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of each A_i is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of each A_i's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by each A_i.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of each matrix A_i.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of each matrix A_i.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device array of device pointers storing each banded triangular matrix A_i.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. lda must satisfy lda > k.
    @param[inout]
    x         device array of device pointer storing each vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   int                               k,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batch_count);

// tbmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                k,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                k,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   k,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         stride_a,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stride_x,
                                                          int                   batch_count);

/*! \brief BLAS Level 2 API

    \details
    xTBMV_STRIDED_BATCHED performs one of the matrix-vector operations

        x_i := A_i*x_i      or
        x_i := A_i**T*x_i   or
        x_i := A_i**H*x_i,

    where (A_i, x_i) is the i-th instance of the batch.
    x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batch_count.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    uplo      [rocblas_fill]
              rocblas_fill_upper: each A_i is an upper banded triangular matrix.
              rocblas_fill_lower: each A_i is a  lower banded triangular matrix.
    @param[in]
    trans     [rocblas_operation]
              indicates whether each matrix A_i is tranposed (conjugated) or not.
    @param[in]
    diag      [rocblas_diagonal]
              rocblas_diagonal_unit: The main diagonal of each A_i is assumed to consist of only
                                     1's and is not referenced.
              rocblas_diagonal_non_unit: No assumptions are made of each A_i's main diagonal.
    @param[in]
    m         [rocblas_int]
              the number of rows and columns of the matrix represented by each A_i.
    @param[in]
    k         [rocblas_int]
              if uplo == rocblas_fill_upper, k specifies the number of super-diagonals
              of each matrix A_i.
              if uplo == rocblas_fill_lower, k specifies the number of sub-diagonals
              of each matrix A_i.
              k must satisfy k > 0 && k < lda.
    @param[in]
    A         device array to the first matrix A_i of the batch. Stores each banded triangular matrix A_i.
              if uplo == rocblas_fill_upper:
                The matrix represented is an upper banded triangular matrix
                with the main diagonal and k super-diagonals, everything
                else can be assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the k'th
                row, the first super diagonal resides on the RHS of the k-1'th row, etc,
                with the k'th diagonal on the RHS of the 0'th row.
                   Ex: (rocblas_fill_upper; m = 5; k = 2)
                      1 6 9 0 0              0 0 9 8 7
                      0 2 7 8 0              0 6 7 8 9
                      0 0 3 8 7     ---->    1 2 3 4 5
                      0 0 0 4 9              0 0 0 0 0
                      0 0 0 0 5              0 0 0 0 0
              if uplo == rocblas_fill_lower:
                The matrix represnted is a lower banded triangular matrix
                with the main diagonal and k sub-diagonals, everything else can be
                assumed to be 0.
                The matrix is compacted so that the main diagonal resides on the 0'th row,
                working up to the k'th diagonal residing on the LHS of the k'th row.
                   Ex: (rocblas_fill_lower; m = 5; k = 2)
                      1 0 0 0 0              1 2 3 4 5
                      6 2 0 0 0              6 7 8 9 0
                      9 7 3 0 0     ---->    9 8 7 0 0
                      0 8 8 4 0              0 0 0 0 0
                      0 0 7 9 5              0 0 0 0 0
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i. lda must satisfy lda > k.
    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_(i + 1).
    @param[inout]
    x         device array to the first vector x_i of the batch.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i matrix to the next x_(i + 1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          int                         k,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               stride_a,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stride_x,
                                                          int                         batch_count);

// tbsv
HIPBLAS_EXPORT hipblasStatus_t hipblasStbsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                n,
                                            int                k,
                                            const float*       A,
                                            int                lda,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                n,
                                            int                k,
                                            const double*      A,
                                            int                lda,
                                            double*            x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    tbsv solves

         A*x = b or A**T*x = b or A**H*x = b,

    where x and b are vectors and A is a banded triangular matrix.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A*x = b
               rocblas_operation_transpose: Solves A**T*x = b
               rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular (i.e. the diagonal elements
                                       of A are not used in computations).
            rocblas_diagonal_non_unit: A is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of b. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of A.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of A.
              k >= 0.

    @param[in]
    A         device pointer storing the matrix A in banded format.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda >= (k + 1).

    @param[inout]
    x         device pointer storing input vector b. Overwritten by the output vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

// tbsv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStbsvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                n,
                                                   int                k,
                                                   const float* const A[],
                                                   int                lda,
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 n,
                                                   int                 k,
                                                   const double* const A[],
                                                   int                 lda,
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    tbsv_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A_i*x_i = b_i
               rocblas_operation_transpose: Solves A_i**T*x_i = b_i
               rocblas_operation_conjugate_transpose: Solves A_i**H*x_i = b_i

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of each A_i.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of each A_i.
              k >= 0.

    @param[in]
    A         device vector of device pointers storing each matrix A_i in banded format.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda >= (k + 1).

    @param[inout]
    x         device vector of device pointers storing each input vector b_i. Overwritten by each output
              vector x_i.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               n,
                                                   int                               k,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batchCount);

// tbsv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                n,
                                                          int                k,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                n,
                                                          int                k,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    tbsv_strided_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a banded triangular matrix,
    for i = [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]
               rocblas_operation_none: Solves A_i*x_i = b_i
               rocblas_operation_transpose: Solves A_i**T*x_i = b_i
               rocblas_operation_conjugate_transpose: Solves A_i**H*x_i = b_i

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.
    @param[in]
    k         [rocblas_int]
              if(uplo == rocblas_fill_upper)
                k specifies the number of super-diagonals of each A_i.
              if(uplo == rocblas_fill_lower)
                k specifies the number of sub-diagonals of each A_i.
              k >= 0.

    @param[in]
    A         device pointer pointing to the first banded matrix A_1.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda >= (k + 1).
    @param[in]
    stride_A  [rocblas_stride]
              specifies the distance between the start of one matrix (A_i) and the next (A_i+1).

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by output vectors x.
    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              specifies the distance between the start of one vector (x_i) and the next (x_i+1).
    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         n,
                                                          int                         k,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          int                         batchCount);

// tpmv
HIPBLAS_EXPORT hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const float*       AP,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const double*      AP,
                                            double*            x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* AP,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    tpmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.

    The vector x is overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m       [rocblas_int]
            m specifies the number of rows of A. m >= 0.

    @param[in]
    A       device pointer storing matrix A,
            of dimension at leat ( m * ( m + 1 ) / 2 ).
          Before entry with uplo = rocblas_fill_upper, the array A
          must contain the upper triangular matrix packed sequentially,
          column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
          a_{0,1} and a_{1, 1} respectively, and so on.
          Before entry with uplo = rocblas_fill_lower, the array A
          must contain the lower triangular matrix packed sequentially,
          column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
          a_{1,0} and a_{2,0} respectively, and so on.
          Note that when DIAG = rocblas_diagonal_unit, the diagonal elements of A are
          not referenced, but are assumed to be unity.

    @param[in]
    x       device pointer storing vector x.

    @param[in]
    incx    [rocblas_int]
            specifies the increment for the elements of x. incx must not be zero.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            const hipblasDoubleComplex* AP,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

// tpmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStpmvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   const float* const AP[],
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   const double* const AP[],
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const AP[],
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    tpmv_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer storing pointer of matrices A_i,
              of dimension ( lda, m )

    @param[in]
    x         device pointer storing vectors x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of vectors x_i.

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   const hipblasDoubleComplex* const AP[],
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batchCount);

// tpmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const float*       AP,
                                                          hipblasStride      strideAP,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stride,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const double*      AP,
                                                          hipblasStride      strideAP,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stride,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* AP,
                                                          hipblasStride         strideAP,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stride,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    tpmv_strided_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer of the matrix A_0,
              of dimension ( lda, m )

    @param[in]
    stride_a  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_{i + 1}

    @param[in]
    x         device pointer storing the vector x_0.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of one vector x.

    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i vector to the next x_{i + 1}

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          const hipblasDoubleComplex* AP,
                                                          hipblasStride               strideAP,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stride,
                                                          int                         batchCount);

// tpsv
HIPBLAS_EXPORT hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const float*       AP,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const double*      AP,
                                            double*            x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* AP,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    TPSV solves

         A*x = b or A**T*x = b, or A**H*x = b,

    where x and b are vectors and A is a triangular matrix stored in the packed format.

    The input vector b is overwritten by the output vector x.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular (i.e. the diagonal elements
                                       of A are not used in computations).
            rocblas_diagonal_non_unit: A is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of b. n >= 0.

    @param[in]
    AP        device pointer storing the packed version of matrix A,
              of dimension >= (n * (n + 1) / 2)

    @param[inout]
    x         device pointer storing vector b on input, overwritten by x on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            const hipblasDoubleComplex* AP,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

// tpsv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStpsvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   const float* const AP[],
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   const double* const AP[],
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const AP[],
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batchCount);

/*! \brief BLAS Level 2 API

    \details
    TPSV_BATCHED solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.

    @param[in]
    AP        device array of device pointers storing the packed versions of each matrix A_i,
              of dimension >= (n * (n + 1) / 2)

    @param[inout]
    x         device array of device pointers storing each input vector b_i, overwritten by x_i on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   const hipblasDoubleComplex* const AP[],
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batchCount);

// tpsv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const float*       AP,
                                                          hipblasStride      strideAP,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const double*      AP,
                                                          hipblasStride      strideAP,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* AP,
                                                          hipblasStride         strideAP,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          int                   batchCount);

/*! \brief BLAS Level 2 API

    \details
    TPSV_STRIDED_BATCHED solves

         A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,

    where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
    for i in [1, batch_count].

    The input vectors b_i are overwritten by the output vectors x_i.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  each A_i is an upper triangular matrix.
            rocblas_fill_lower:  each A_i is a  lower triangular matrix.

    @param[in]
    transA  [rocblas_operation]
            rocblas_operation_none: Solves A*x = b
            rocblas_operation_transpose: Solves A**T*x = b
            rocblas_operation_conjugate_transpose: Solves A**H*x = b

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
                                       of each A_i are not used in computations).
            rocblas_diagonal_non_unit: each A_i is not assumed to be unit triangular.

    @param[in]
    n         [rocblas_int]
              n specifies the number of rows of each b_i. n >= 0.

    @param[in]
    AP        device pointer pointing to the first packed matrix A_1,
              of dimension >= (n * (n + 1) / 2)

    @param[in]
    stride_A  [rocblas_stride]
              stride from the beginning of one packed matrix (AP_i) and the next (AP_i+1).

    @param[inout]
    x         device pointer pointing to the first input vector b_1. Overwritten by each x_i on output.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.
    @param[in]
    stride_x  [rocblas_stride]
              stride from the beginning of one vector (x_i) and the next (x_i+1).
    @param[in]
    batch_count [rocblas_int]
                specifies the number of instances in the batch.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          const hipblasDoubleComplex* AP,
                                                          hipblasStride               strideAP,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          int                         batchCount);

// trmv
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const float*       A,
                                            int                lda,
                                            float*             x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmv(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            hipblasDiagType_t  diag,
                                            int                m,
                                            const double*      A,
                                            int                lda,
                                            double*            x,
                                            int                incx);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    trmv performs one of the matrix-vector operations

         x = A*x or x = A**T*x,

    where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.

    The vector x is overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of A. m >= 0.

    @param[in]
    A         device pointer storing matrix A,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         device pointer storing vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

// trmv_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmvBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   const float* const A[],
                                                   int                lda,
                                                   float* const       x[],
                                                   int                incx,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   const double* const A[],
                                                   int                 lda,
                                                   double* const       x[],
                                                   int                 incx,
                                                   int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count);

/*! \brief BLAS Level 2 API

    \details
    trmv_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer storing pointer of matrices A_i,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A_i.
              lda >= max( 1, m ).

    @param[in]
    x         device pointer storing vectors x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of vectors x_i.

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batch_count);

// trmv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      stride_a,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stride_x,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         stride_a,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stride_x,
                                                          int                   batch_count);

/*! \brief BLAS Level 2 API

    \details
    trmv_strided_batched performs one of the matrix-vector operations

         x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batch_count

    where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
    with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).

    The vectors x_i are overwritten.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A_i is an upper triangular matrix.
            rocblas_fill_lower:  A_i is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A_i is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A_i is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of matrices A_i. m >= 0.

    @param[in]
    A         device pointer of the matrix A_0,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A_i.
              lda >= max( 1, m ).

    @param[in]
    stride_a  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_{i + 1}

    @param[in]
    x         device pointer storing the vector x_0.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of one vector x.

    @param[in]
    stride_x  [rocblas_stride]
              stride from the start of one x_i vector to the next x_{i + 1}

    @param[in]
    batch_count [rocblas_int]
              The number of batched matrices/vectors.


    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               stride_a,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stride_x,
                                                          int                         batch_count);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       x,
                                            int                   incx);

/*! \brief BLAS Level 2 API

    \details
    trsv solves

         A*x = b or A**T*x = b,

    where x and b are vectors and A is a triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         device pointer storing matrix A,
              of dimension ( lda, m )

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
              lda = max( 1, m ).

    @param[in]
    x         device pointer storing vector x.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasDoubleComplex*       x,
                                            int                         incx);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       x[],
                                                   int                         incx,
                                                   int                         batch_count);

/*! \brief BLAS Level 2 API

    \details
    trsv_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an
    m by m triangular matrix.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of b. m >= 0.

    @param[in]
    A         device array of device pointers storing each matrix A_i.

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda = max(1, m)

    @param[in]
    x         device array of device pointers storing each vector x_i.

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of x.

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   hipblasDoubleComplex* const       x[],
                                                   int                               incx,
                                                   int                               batch_count);

// trsv_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          float*             x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          double*            x,
                                                          int                incx,
                                                          hipblasStride      stridex,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          hipblasComplex*       x,
                                                          int                   incx,
                                                          hipblasStride         stridex,
                                                          int                   batch_count);

/*! \brief BLAS Level 2 API

    \details
    trsv_strided_batched solves

         A_i*x_i = b_i or A_i**T*x_i = b_i,

    where (A_i, x_i, b_i) is the i-th instance of the batch.
    x_i and b_i are vectors and A_i is an m by m triangular matrix, for i = 1, ..., batch_count.

    The vector x is overwritten on b.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.

    @param[in]
    uplo    [rocblas_fill]
            rocblas_fill_upper:  A is an upper triangular matrix.
            rocblas_fill_lower:  A is a  lower triangular matrix.

    @param[in]
    transA     [rocblas_operation]

    @param[in]
    diag    [rocblas_diagonal]
            rocblas_diagonal_unit:     A is assumed to be unit triangular.
            rocblas_diagonal_non_unit:  A is not assumed to be unit triangular.

    @param[in]
    m         [rocblas_int]
              m specifies the number of rows of each b_i. m >= 0.

    @param[in]
    A         device pointer to the first matrix (A_1) in the batch, of dimension ( lda, m )

    @param[in]
    stride_A  [rocblas_stride]
              stride from the start of one A_i matrix to the next A_(i + 1)

    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of each A_i.
              lda = max( 1, m ).

    @param[in, out]
    x         device pointer to the first vector (x_1) in the batch.

    @param[in]
    stride_x [rocblas_stride]
             stride from the start of one x_i vector to the next x_(i + 1)

    @param[in]
    incx      [rocblas_int]
              specifies the increment for the elements of each x_i.

    @param[in]
    batch_count [rocblas_int]
                number of instances in the batch

    ********************************************************************/
HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          hipblasDoubleComplex*       x,
                                                          int                         incx,
                                                          hipblasStride               stridex,
                                                          int                         batch_count);

// ================================
// ========== LEVEL 3 =============
// ================================

// herk
HIPBLAS_EXPORT hipblasStatus_t hipblasCherk(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const float*          alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherk(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const double*               alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// herk_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherkBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const float*                alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const float*                beta,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherkBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   int                               n,
                                                   int                               k,
                                                   const double*                     alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const double*                     beta,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batchCount);

// herk_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherkStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          int                   n,
                                                          int                   k,
                                                          const float*          alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const float*          beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherkStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          int                         n,
                                                          int                         k,
                                                          const double*               alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const double*               beta,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               strideC,
                                                          int                         batchCount);

// herkx
HIPBLAS_EXPORT hipblasStatus_t hipblasCherkx(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasOperation_t    transA,
                                             int                   n,
                                             int                   k,
                                             const hipblasComplex* alpha,
                                             const hipblasComplex* A,
                                             int                   lda,
                                             const hipblasComplex* B,
                                             int                   ldb,
                                             const float*          beta,
                                             hipblasComplex*       C,
                                             int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherkx(hipblasHandle_t             handle,
                                             hipblasFillMode_t           uplo,
                                             hipblasOperation_t          transA,
                                             int                         n,
                                             int                         k,
                                             const hipblasDoubleComplex* alpha,
                                             const hipblasDoubleComplex* A,
                                             int                         lda,
                                             const hipblasDoubleComplex* B,
                                             int                         ldb,
                                             const double*               beta,
                                             hipblasDoubleComplex*       C,
                                             int                         ldc);

// herkx_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherkxBatched(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasOperation_t          transA,
                                                    int                         n,
                                                    int                         k,
                                                    const hipblasComplex*       alpha,
                                                    const hipblasComplex* const A[],
                                                    int                         lda,
                                                    const hipblasComplex* const B[],
                                                    int                         ldb,
                                                    const float*                beta,
                                                    hipblasComplex* const       C[],
                                                    int                         ldc,
                                                    int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherkxBatched(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasOperation_t                transA,
                                                    int                               n,
                                                    int                               k,
                                                    const hipblasDoubleComplex*       alpha,
                                                    const hipblasDoubleComplex* const A[],
                                                    int                               lda,
                                                    const hipblasDoubleComplex* const B[],
                                                    int                               ldb,
                                                    const double*                     beta,
                                                    hipblasDoubleComplex* const       C[],
                                                    int                               ldc,
                                                    int                               batchCount);

// herkx_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCherkxStridedBatched(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasOperation_t    transA,
                                                           int                   n,
                                                           int                   k,
                                                           const hipblasComplex* alpha,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         strideA,
                                                           const hipblasComplex* B,
                                                           int                   ldb,
                                                           hipblasStride         strideB,
                                                           const float*          beta,
                                                           hipblasComplex*       C,
                                                           int                   ldc,
                                                           hipblasStride         strideC,
                                                           int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZherkxStridedBatched(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasOperation_t          transA,
                                                           int                         n,
                                                           int                         k,
                                                           const hipblasDoubleComplex* alpha,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               strideA,
                                                           const hipblasDoubleComplex* B,
                                                           int                         ldb,
                                                           hipblasStride               strideB,
                                                           const double*               beta,
                                                           hipblasDoubleComplex*       C,
                                                           int                         ldc,
                                                           hipblasStride               strideC,
                                                           int                         batchCount);

// her2k
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2k(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasOperation_t    transA,
                                             int                   n,
                                             int                   k,
                                             const hipblasComplex* alpha,
                                             const hipblasComplex* A,
                                             int                   lda,
                                             const hipblasComplex* B,
                                             int                   ldb,
                                             const float*          beta,
                                             hipblasComplex*       C,
                                             int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZher2k(hipblasHandle_t             handle,
                                             hipblasFillMode_t           uplo,
                                             hipblasOperation_t          transA,
                                             int                         n,
                                             int                         k,
                                             const hipblasDoubleComplex* alpha,
                                             const hipblasDoubleComplex* A,
                                             int                         lda,
                                             const hipblasDoubleComplex* B,
                                             int                         ldb,
                                             const double*               beta,
                                             hipblasDoubleComplex*       C,
                                             int                         ldc);

// her2k_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2kBatched(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasOperation_t          transA,
                                                    int                         n,
                                                    int                         k,
                                                    const hipblasComplex*       alpha,
                                                    const hipblasComplex* const A[],
                                                    int                         lda,
                                                    const hipblasComplex* const B[],
                                                    int                         ldb,
                                                    const float*                beta,
                                                    hipblasComplex* const       C[],
                                                    int                         ldc,
                                                    int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZher2kBatched(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasOperation_t                transA,
                                                    int                               n,
                                                    int                               k,
                                                    const hipblasDoubleComplex*       alpha,
                                                    const hipblasDoubleComplex* const A[],
                                                    int                               lda,
                                                    const hipblasDoubleComplex* const B[],
                                                    int                               ldb,
                                                    const double*                     beta,
                                                    hipblasDoubleComplex* const       C[],
                                                    int                               ldc,
                                                    int                               batchCount);

// her2k_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasCher2kStridedBatched(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasOperation_t    transA,
                                                           int                   n,
                                                           int                   k,
                                                           const hipblasComplex* alpha,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         strideA,
                                                           const hipblasComplex* B,
                                                           int                   ldb,
                                                           hipblasStride         strideB,
                                                           const float*          beta,
                                                           hipblasComplex*       C,
                                                           int                   ldc,
                                                           hipblasStride         strideC,
                                                           int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZher2kStridedBatched(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasOperation_t          transA,
                                                           int                         n,
                                                           int                         k,
                                                           const hipblasDoubleComplex* alpha,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               strideA,
                                                           const hipblasDoubleComplex* B,
                                                           int                         ldb,
                                                           hipblasStride               strideB,
                                                           const double*               beta,
                                                           hipblasDoubleComplex*       C,
                                                           int                         ldc,
                                                           hipblasStride               strideC,
                                                           int                         batchCount);

// symm
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymm(hipblasHandle_t   handle,
                                            hipblasSideMode_t side,
                                            hipblasFillMode_t uplo,
                                            int               m,
                                            int               n,
                                            const float*      alpha,
                                            const float*      A,
                                            int               lda,
                                            const float*      B,
                                            int               ldb,
                                            const float*      beta,
                                            float*            C,
                                            int               ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymm(hipblasHandle_t   handle,
                                            hipblasSideMode_t side,
                                            hipblasFillMode_t uplo,
                                            int               m,
                                            int               n,
                                            const double*     alpha,
                                            const double*     A,
                                            int               lda,
                                            const double*     B,
                                            int               ldb,
                                            const double*     beta,
                                            double*           C,
                                            int               ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymm(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            hipblasFillMode_t     uplo,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsymm(hipblasHandle_t             handle,
                                            hipblasSideMode_t           side,
                                            hipblasFillMode_t           uplo,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// symm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymmBatched(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   int                m,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const B[],
                                                   int                ldb,
                                                   const float*       beta,
                                                   float* const       C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymmBatched(hipblasHandle_t     handle,
                                                   hipblasSideMode_t   side,
                                                   hipblasFillMode_t   uplo,
                                                   int                 m,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const B[],
                                                   int                 ldb,
                                                   const double*       beta,
                                                   double* const       C[],
                                                   int                 ldc,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymmBatched(hipblasHandle_t             handle,
                                                   hipblasSideMode_t           side,
                                                   hipblasFillMode_t           uplo,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const B[],
                                                   int                         ldb,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsymmBatched(hipblasHandle_t                   handle,
                                                   hipblasSideMode_t                 side,
                                                   hipblasFillMode_t                 uplo,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const B[],
                                                   int                               ldb,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batchCount);

// symm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsymmStridedBatched(hipblasHandle_t   handle,
                                                          hipblasSideMode_t side,
                                                          hipblasFillMode_t uplo,
                                                          int               m,
                                                          int               n,
                                                          const float*      alpha,
                                                          const float*      A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const float*      B,
                                                          int               ldb,
                                                          hipblasStride     strideB,
                                                          const float*      beta,
                                                          float*            C,
                                                          int               ldc,
                                                          hipblasStride     strideC,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsymmStridedBatched(hipblasHandle_t   handle,
                                                          hipblasSideMode_t side,
                                                          hipblasFillMode_t uplo,
                                                          int               m,
                                                          int               n,
                                                          const double*     alpha,
                                                          const double*     A,
                                                          int               lda,
                                                          hipblasStride     strideA,
                                                          const double*     B,
                                                          int               ldb,
                                                          hipblasStride     strideB,
                                                          const double*     beta,
                                                          double*           C,
                                                          int               ldc,
                                                          hipblasStride     strideC,
                                                          int               batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsymmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsymmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasSideMode_t           side,
                                                          hipblasFillMode_t           uplo,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* B,
                                                          int                         ldb,
                                                          hipblasStride               strideB,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               strideC,
                                                          int                         batchCount);

// syrk
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrk(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrk(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrk(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrk(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// syrk_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkBatched(hipblasHandle_t    handle,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   int                n,
                                                   int                k,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float*       beta,
                                                   float* const       C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkBatched(hipblasHandle_t     handle,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   int                 n,
                                                   int                 k,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double*       beta,
                                                   double* const       C[],
                                                   int                 ldc,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkBatched(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int                         n,
                                                   int                         k,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex*       beta,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkBatched(hipblasHandle_t                   handle,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   int                               n,
                                                   int                               k,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex*       beta,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batchCount);

// syrk_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          int                n,
                                                          int                k,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          const float*       beta,
                                                          float*             C,
                                                          int                ldc,
                                                          hipblasStride      strideC,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkStridedBatched(hipblasHandle_t    handle,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          int                n,
                                                          int                k,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          const double*      beta,
                                                          double*            C,
                                                          int                ldc,
                                                          hipblasStride      strideC,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkStridedBatched(hipblasHandle_t       handle,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkStridedBatched(hipblasHandle_t             handle,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          int                         n,
                                                          int                         k,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               strideC,
                                                          int                         batchCount);

// syr2k
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2k(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2k(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2k(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasOperation_t    transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2k(hipblasHandle_t             handle,
                                             hipblasFillMode_t           uplo,
                                             hipblasOperation_t          transA,
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

// syr2k_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2kBatched(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2kBatched(hipblasHandle_t     handle,
                                                    hipblasFillMode_t   uplo,
                                                    hipblasOperation_t  transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2kBatched(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasOperation_t          transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2kBatched(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasOperation_t                transA,
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

// syr2k_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2kStridedBatched(hipblasHandle_t    handle,
                                                           hipblasFillMode_t  uplo,
                                                           hipblasOperation_t transA,
                                                           int                n,
                                                           int                k,
                                                           const float*       alpha,
                                                           const float*       A,
                                                           int                lda,
                                                           hipblasStride      strideA,
                                                           const float*       B,
                                                           int                ldb,
                                                           hipblasStride      strideB,
                                                           const float*       beta,
                                                           float*             C,
                                                           int                ldc,
                                                           hipblasStride      strideC,
                                                           int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2kStridedBatched(hipblasHandle_t    handle,
                                                           hipblasFillMode_t  uplo,
                                                           hipblasOperation_t transA,
                                                           int                n,
                                                           int                k,
                                                           const double*      alpha,
                                                           const double*      A,
                                                           int                lda,
                                                           hipblasStride      strideA,
                                                           const double*      B,
                                                           int                ldb,
                                                           hipblasStride      strideB,
                                                           const double*      beta,
                                                           double*            C,
                                                           int                ldc,
                                                           hipblasStride      strideC,
                                                           int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2kStridedBatched(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasOperation_t    transA,
                                                           int                   n,
                                                           int                   k,
                                                           const hipblasComplex* alpha,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         strideA,
                                                           const hipblasComplex* B,
                                                           int                   ldb,
                                                           hipblasStride         strideB,
                                                           const hipblasComplex* beta,
                                                           hipblasComplex*       C,
                                                           int                   ldc,
                                                           hipblasStride         strideC,
                                                           int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2kStridedBatched(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasOperation_t          transA,
                                                           int                         n,
                                                           int                         k,
                                                           const hipblasDoubleComplex* alpha,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               strideA,
                                                           const hipblasDoubleComplex* B,
                                                           int                         ldb,
                                                           hipblasStride               strideB,
                                                           const hipblasDoubleComplex* beta,
                                                           hipblasDoubleComplex*       C,
                                                           int                         ldc,
                                                           hipblasStride               strideC,
                                                           int                         batchCount);

// syrkx
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkx(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkx(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkx(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasOperation_t    transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkx(hipblasHandle_t             handle,
                                             hipblasFillMode_t           uplo,
                                             hipblasOperation_t          transA,
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

// syrkx_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkxBatched(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkxBatched(hipblasHandle_t     handle,
                                                    hipblasFillMode_t   uplo,
                                                    hipblasOperation_t  transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkxBatched(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasOperation_t          transA,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkxBatched(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasOperation_t                transA,
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

// syrkx_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkxStridedBatched(hipblasHandle_t    handle,
                                                           hipblasFillMode_t  uplo,
                                                           hipblasOperation_t transA,
                                                           int                n,
                                                           int                k,
                                                           const float*       alpha,
                                                           const float*       A,
                                                           int                lda,
                                                           hipblasStride      strideA,
                                                           const float*       B,
                                                           int                ldb,
                                                           hipblasStride      strideB,
                                                           const float*       beta,
                                                           float*             C,
                                                           int                ldc,
                                                           hipblasStride      stridec,
                                                           int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkxStridedBatched(hipblasHandle_t    handle,
                                                           hipblasFillMode_t  uplo,
                                                           hipblasOperation_t transA,
                                                           int                n,
                                                           int                k,
                                                           const double*      alpha,
                                                           const double*      A,
                                                           int                lda,
                                                           hipblasStride      strideA,
                                                           const double*      B,
                                                           int                ldb,
                                                           hipblasStride      strideB,
                                                           const double*      beta,
                                                           double*            C,
                                                           int                ldc,
                                                           hipblasStride      stridec,
                                                           int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkxStridedBatched(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasOperation_t    transA,
                                                           int                   n,
                                                           int                   k,
                                                           const hipblasComplex* alpha,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         strideA,
                                                           const hipblasComplex* B,
                                                           int                   ldb,
                                                           hipblasStride         strideB,
                                                           const hipblasComplex* beta,
                                                           hipblasComplex*       C,
                                                           int                   ldc,
                                                           hipblasStride         stridec,
                                                           int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkxStridedBatched(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasOperation_t          transA,
                                                           int                         n,
                                                           int                         k,
                                                           const hipblasDoubleComplex* alpha,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               strideA,
                                                           const hipblasDoubleComplex* B,
                                                           int                         ldb,
                                                           hipblasStride               strideB,
                                                           const hipblasDoubleComplex* beta,
                                                           hipblasDoubleComplex*       C,
                                                           int                         ldc,
                                                           hipblasStride               stridec,
                                                           int                         batchCount);

// geam
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeam(hipblasHandle_t       handle,
                                            hipblasOperation_t    transa,
                                            hipblasOperation_t    transb,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* beta,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeam(hipblasHandle_t             handle,
                                            hipblasOperation_t          transa,
                                            hipblasOperation_t          transb,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* beta,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// geam_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgeamBatched(hipblasHandle_t    handle,
                                                   hipblasOperation_t transa,
                                                   hipblasOperation_t transb,
                                                   int                m,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float*       beta,
                                                   const float* const B[],
                                                   int                ldb,
                                                   float* const       C[],
                                                   int                ldc,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeamBatched(hipblasHandle_t     handle,
                                                   hipblasOperation_t  transa,
                                                   hipblasOperation_t  transb,
                                                   int                 m,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double*       beta,
                                                   const double* const B[],
                                                   int                 ldb,
                                                   double* const       C[],
                                                   int                 ldc,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeamBatched(hipblasHandle_t             handle,
                                                   hipblasOperation_t          transa,
                                                   hipblasOperation_t          transb,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex*       beta,
                                                   const hipblasComplex* const B[],
                                                   int                         ldb,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeamBatched(hipblasHandle_t                   handle,
                                                   hipblasOperation_t                transa,
                                                   hipblasOperation_t                transb,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex*       beta,
                                                   const hipblasDoubleComplex* const B[],
                                                   int                               ldb,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batchCount);

// geam_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgeamStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t transa,
                                                          hipblasOperation_t transb,
                                                          int                m,
                                                          int                n,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          const float*       beta,
                                                          const float*       B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
                                                          float*             C,
                                                          int                ldc,
                                                          hipblasStride      strideC,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeamStridedBatched(hipblasHandle_t    handle,
                                                          hipblasOperation_t transa,
                                                          hipblasOperation_t transb,
                                                          int                m,
                                                          int                n,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          const double*      beta,
                                                          const double*      B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
                                                          double*            C,
                                                          int                ldc,
                                                          hipblasStride      strideC,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeamStridedBatched(hipblasHandle_t       handle,
                                                          hipblasOperation_t    transa,
                                                          hipblasOperation_t    transb,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* beta,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeamStridedBatched(hipblasHandle_t             handle,
                                                          hipblasOperation_t          transa,
                                                          hipblasOperation_t          transb,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* beta,
                                                          const hipblasDoubleComplex* B,
                                                          int                         ldb,
                                                          hipblasStride               strideB,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               strideC,
                                                          int                         batchCount);

// hemm
HIPBLAS_EXPORT hipblasStatus_t hipblasChemm(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            hipblasFillMode_t     uplo,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZhemm(hipblasHandle_t             handle,
                                            hipblasSideMode_t           side,
                                            hipblasFillMode_t           uplo,
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

// hemm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChemmBatched(hipblasHandle_t             handle,
                                                   hipblasSideMode_t           side,
                                                   hipblasFillMode_t           uplo,
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

HIPBLAS_EXPORT hipblasStatus_t hipblasZhemmBatched(hipblasHandle_t                   handle,
                                                   hipblasSideMode_t                 side,
                                                   hipblasFillMode_t                 uplo,
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

// hemm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasChemmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          int                   n,
                                                          int                   k,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          const hipblasComplex* B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          const hipblasComplex* beta,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         strideC,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZhemmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasSideMode_t           side,
                                                          hipblasFillMode_t           uplo,
                                                          int                         n,
                                                          int                         k,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          const hipblasDoubleComplex* B,
                                                          int                         ldb,
                                                          hipblasStride               strideB,
                                                          const hipblasDoubleComplex* beta,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               strideC,
                                                          int                         batchCount);

// clang-format off
HIPBLAS_DEPRECATED_MSG("The hipblasXtrmm API, along with batched versions, will \
be changing in a future release to allow in-place and out-of-place behavior. This change \
will introduce an output matrix 'C', matching the rocblas_xtrmm_outofplace API and the \
cublasXtrmm API.")
// clang-format on
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

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmm(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasComplex*       B,
                                            int                   ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmm(hipblasHandle_t             handle,
                                            hipblasSideMode_t           side,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasDoubleComplex*       B,
                                            int                         ldb);

// trmm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmmBatched(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   hipblasFillMode_t  uplo,
                                                   hipblasOperation_t transA,
                                                   hipblasDiagType_t  diag,
                                                   int                m,
                                                   int                n,
                                                   const float*       alpha,
                                                   const float* const A[],
                                                   int                lda,
                                                   float* const       B[],
                                                   int                ldb,
                                                   int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmmBatched(hipblasHandle_t     handle,
                                                   hipblasSideMode_t   side,
                                                   hipblasFillMode_t   uplo,
                                                   hipblasOperation_t  transA,
                                                   hipblasDiagType_t   diag,
                                                   int                 m,
                                                   int                 n,
                                                   const double*       alpha,
                                                   const double* const A[],
                                                   int                 lda,
                                                   double* const       B[],
                                                   int                 ldb,
                                                   int                 batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmmBatched(hipblasHandle_t             handle,
                                                   hipblasSideMode_t           side,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex*       alpha,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   hipblasComplex* const       B[],
                                                   int                         ldb,
                                                   int                         batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmmBatched(hipblasHandle_t                   handle,
                                                   hipblasSideMode_t                 side,
                                                   hipblasFillMode_t                 uplo,
                                                   hipblasOperation_t                transA,
                                                   hipblasDiagType_t                 diag,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex*       alpha,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   hipblasDoubleComplex* const       B[],
                                                   int                               ldb,
                                                   int                               batchCount);

// trmm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrmmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasSideMode_t  side,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                n,
                                                          const float*       alpha,
                                                          const float*       A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          float*             B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmmStridedBatched(hipblasHandle_t    handle,
                                                          hipblasSideMode_t  side,
                                                          hipblasFillMode_t  uplo,
                                                          hipblasOperation_t transA,
                                                          hipblasDiagType_t  diag,
                                                          int                m,
                                                          int                n,
                                                          const double*      alpha,
                                                          const double*      A,
                                                          int                lda,
                                                          hipblasStride      strideA,
                                                          double*            B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
                                                          int                batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          hipblasComplex*       B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          int                   batchCount);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasSideMode_t           side,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          hipblasDoubleComplex*       B,
                                                          int                         ldb,
                                                          hipblasStride               strideB,
                                                          int                         batchCount);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsm(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            hipblasDiagType_t     diag,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* alpha,
                                            hipblasComplex*       A,
                                            int                   lda,
                                            hipblasComplex*       B,
                                            int                   ldb);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsm(hipblasHandle_t             handle,
                                            hipblasSideMode_t           side,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            hipblasDiagType_t           diag,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* alpha,
                                            hipblasDoubleComplex*       A,
                                            int                         lda,
                                            hipblasDoubleComplex*       B,
                                            int                         ldb);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t       handle,
                                                   hipblasSideMode_t     side,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   hipblasDiagType_t     diag,
                                                   int                   m,
                                                   int                   n,
                                                   const hipblasComplex* alpha,
                                                   hipblasComplex* const A[],
                                                   int                   lda,
                                                   hipblasComplex*       B[],
                                                   int                   ldb,
                                                   int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t             handle,
                                                   hipblasSideMode_t           side,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   hipblasDiagType_t           diag,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasDoubleComplex* alpha,
                                                   hipblasDoubleComplex* const A[],
                                                   int                         lda,
                                                   hipblasDoubleComplex*       B[],
                                                   int                         ldb,
                                                   int                         batch_count);

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
                                                          hipblasStride      strideA,
                                                          float*             B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
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
                                                          hipblasStride      strideA,
                                                          double*            B,
                                                          int                ldb,
                                                          hipblasStride      strideB,
                                                          int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          hipblasFillMode_t     uplo,
                                                          hipblasOperation_t    transA,
                                                          hipblasDiagType_t     diag,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* alpha,
                                                          hipblasComplex*       A,
                                                          int                   lda,
                                                          hipblasStride         strideA,
                                                          hipblasComplex*       B,
                                                          int                   ldb,
                                                          hipblasStride         strideB,
                                                          int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasSideMode_t           side,
                                                          hipblasFillMode_t           uplo,
                                                          hipblasOperation_t          transA,
                                                          hipblasDiagType_t           diag,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* alpha,
                                                          hipblasDoubleComplex*       A,
                                                          int                         lda,
                                                          hipblasStride               strideA,
                                                          hipblasDoubleComplex*       B,
                                                          int                         ldb,
                                                          hipblasStride               strideB,
                                                          int                         batch_count);

// trtri
HIPBLAS_EXPORT hipblasStatus_t hipblasStrtri(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             hipblasDiagType_t diag,
                                             int               n,
                                             const float*      A,
                                             int               lda,
                                             float*            invA,
                                             int               ldinvA);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrtri(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             hipblasDiagType_t diag,
                                             int               n,
                                             const double*     A,
                                             int               lda,
                                             double*           invA,
                                             int               ldinvA);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrtri(hipblasHandle_t       handle,
                                             hipblasFillMode_t     uplo,
                                             hipblasDiagType_t     diag,
                                             int                   n,
                                             const hipblasComplex* A,
                                             int                   lda,
                                             hipblasComplex*       invA,
                                             int                   ldinvA);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrtri(hipblasHandle_t             handle,
                                             hipblasFillMode_t           uplo,
                                             hipblasDiagType_t           diag,
                                             int                         n,
                                             const hipblasDoubleComplex* A,
                                             int                         lda,
                                             hipblasDoubleComplex*       invA,
                                             int                         ldinvA);

// trtri_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrtriBatched(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasDiagType_t  diag,
                                                    int                n,
                                                    const float* const A[],
                                                    int                lda,
                                                    float*             invA[],
                                                    int                ldinvA,
                                                    int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrtriBatched(hipblasHandle_t     handle,
                                                    hipblasFillMode_t   uplo,
                                                    hipblasDiagType_t   diag,
                                                    int                 n,
                                                    const double* const A[],
                                                    int                 lda,
                                                    double*             invA[],
                                                    int                 ldinvA,
                                                    int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrtriBatched(hipblasHandle_t             handle,
                                                    hipblasFillMode_t           uplo,
                                                    hipblasDiagType_t           diag,
                                                    int                         n,
                                                    const hipblasComplex* const A[],
                                                    int                         lda,
                                                    hipblasComplex*             invA[],
                                                    int                         ldinvA,
                                                    int                         batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrtriBatched(hipblasHandle_t                   handle,
                                                    hipblasFillMode_t                 uplo,
                                                    hipblasDiagType_t                 diag,
                                                    int                               n,
                                                    const hipblasDoubleComplex* const A[],
                                                    int                               lda,
                                                    hipblasDoubleComplex*             invA[],
                                                    int                               ldinvA,
                                                    int                               batch_count);

// trtri_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasStrtriStridedBatched(hipblasHandle_t   handle,
                                                           hipblasFillMode_t uplo,
                                                           hipblasDiagType_t diag,
                                                           int               n,
                                                           const float*      A,
                                                           int               lda,
                                                           hipblasStride     stride_A,
                                                           float*            invA,
                                                           int               ldinvA,
                                                           hipblasStride     stride_invA,
                                                           int               batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDtrtriStridedBatched(hipblasHandle_t   handle,
                                                           hipblasFillMode_t uplo,
                                                           hipblasDiagType_t diag,
                                                           int               n,
                                                           const double*     A,
                                                           int               lda,
                                                           hipblasStride     stride_A,
                                                           double*           invA,
                                                           int               ldinvA,
                                                           hipblasStride     stride_invA,
                                                           int               batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCtrtriStridedBatched(hipblasHandle_t       handle,
                                                           hipblasFillMode_t     uplo,
                                                           hipblasDiagType_t     diag,
                                                           int                   n,
                                                           const hipblasComplex* A,
                                                           int                   lda,
                                                           hipblasStride         stride_A,
                                                           hipblasComplex*       invA,
                                                           int                   ldinvA,
                                                           hipblasStride         stride_invA,
                                                           int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZtrtriStridedBatched(hipblasHandle_t             handle,
                                                           hipblasFillMode_t           uplo,
                                                           hipblasDiagType_t           diag,
                                                           int                         n,
                                                           const hipblasDoubleComplex* A,
                                                           int                         lda,
                                                           hipblasStride               stride_A,
                                                           hipblasDoubleComplex*       invA,
                                                           int                         ldinvA,
                                                           hipblasStride               stride_invA,
                                                           int                         batch_count);

// dgmm
HIPBLAS_EXPORT hipblasStatus_t hipblasSdgmm(hipblasHandle_t   handle,
                                            hipblasSideMode_t side,
                                            int               m,
                                            int               n,
                                            const float*      A,
                                            int               lda,
                                            const float*      x,
                                            int               incx,
                                            float*            C,
                                            int               ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdgmm(hipblasHandle_t   handle,
                                            hipblasSideMode_t side,
                                            int               m,
                                            int               n,
                                            const double*     A,
                                            int               lda,
                                            const double*     x,
                                            int               incx,
                                            double*           C,
                                            int               ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmm(hipblasHandle_t       handle,
                                            hipblasSideMode_t     side,
                                            int                   m,
                                            int                   n,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasComplex*       C,
                                            int                   ldc);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmm(hipblasHandle_t             handle,
                                            hipblasSideMode_t           side,
                                            int                         m,
                                            int                         n,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc);

// dgmm_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSdgmmBatched(hipblasHandle_t    handle,
                                                   hipblasSideMode_t  side,
                                                   int                m,
                                                   int                n,
                                                   const float* const A[],
                                                   int                lda,
                                                   const float* const x[],
                                                   int                incx,
                                                   float* const       C[],
                                                   int                ldc,
                                                   int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdgmmBatched(hipblasHandle_t     handle,
                                                   hipblasSideMode_t   side,
                                                   int                 m,
                                                   int                 n,
                                                   const double* const A[],
                                                   int                 lda,
                                                   const double* const x[],
                                                   int                 incx,
                                                   double* const       C[],
                                                   int                 ldc,
                                                   int                 batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmmBatched(hipblasHandle_t             handle,
                                                   hipblasSideMode_t           side,
                                                   int                         m,
                                                   int                         n,
                                                   const hipblasComplex* const A[],
                                                   int                         lda,
                                                   const hipblasComplex* const x[],
                                                   int                         incx,
                                                   hipblasComplex* const       C[],
                                                   int                         ldc,
                                                   int                         batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmmBatched(hipblasHandle_t                   handle,
                                                   hipblasSideMode_t                 side,
                                                   int                               m,
                                                   int                               n,
                                                   const hipblasDoubleComplex* const A[],
                                                   int                               lda,
                                                   const hipblasDoubleComplex* const x[],
                                                   int                               incx,
                                                   hipblasDoubleComplex* const       C[],
                                                   int                               ldc,
                                                   int                               batch_count);

// dgmm_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSdgmmStridedBatched(hipblasHandle_t   handle,
                                                          hipblasSideMode_t side,
                                                          int               m,
                                                          int               n,
                                                          const float*      A,
                                                          int               lda,
                                                          hipblasStride     stride_A,
                                                          const float*      x,
                                                          int               incx,
                                                          hipblasStride     stride_x,
                                                          float*            C,
                                                          int               ldc,
                                                          hipblasStride     stride_C,
                                                          int               batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDdgmmStridedBatched(hipblasHandle_t   handle,
                                                          hipblasSideMode_t side,
                                                          int               m,
                                                          int               n,
                                                          const double*     A,
                                                          int               lda,
                                                          hipblasStride     stride_A,
                                                          const double*     x,
                                                          int               incx,
                                                          hipblasStride     stride_x,
                                                          double*           C,
                                                          int               ldc,
                                                          hipblasStride     stride_C,
                                                          int               batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmmStridedBatched(hipblasHandle_t       handle,
                                                          hipblasSideMode_t     side,
                                                          int                   m,
                                                          int                   n,
                                                          const hipblasComplex* A,
                                                          int                   lda,
                                                          hipblasStride         stride_A,
                                                          const hipblasComplex* x,
                                                          int                   incx,
                                                          hipblasStride         stride_x,
                                                          hipblasComplex*       C,
                                                          int                   ldc,
                                                          hipblasStride         stride_C,
                                                          int                   batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmmStridedBatched(hipblasHandle_t             handle,
                                                          hipblasSideMode_t           side,
                                                          int                         m,
                                                          int                         n,
                                                          const hipblasDoubleComplex* A,
                                                          int                         lda,
                                                          hipblasStride               stride_A,
                                                          const hipblasDoubleComplex* x,
                                                          int                         incx,
                                                          hipblasStride               stride_x,
                                                          hipblasDoubleComplex*       C,
                                                          int                         ldc,
                                                          hipblasStride               stride_C,
                                                          int                         batch_count);

// getrf
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrf(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrf(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrf(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrf(hipblasHandle_t       handle,
                                             const int             n,
                                             hipblasDoubleComplex* A,
                                             const int             lda,
                                             int*                  ipiv,
                                             int*                  info);

// getrf_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle,
                                                    const int       n,
                                                    float* const    A[],
                                                    const int       lda,
                                                    int*            ipiv,
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle,
                                                    const int       n,
                                                    double* const   A[],
                                                    const int       lda,
                                                    int*            ipiv,
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrfBatched(hipblasHandle_t       handle,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    int*                  ipiv,
                                                    int*                  info,
                                                    const int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t             handle,
                                                    const int                   n,
                                                    hipblasDoubleComplex* const A[],
                                                    const int                   lda,
                                                    int*                        ipiv,
                                                    int*                        info,
                                                    const int                   batch_count);

// getrf_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           n,
                                                           float*              A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           int*                ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           n,
                                                           double*             A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           int*                ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           n,
                                                           hipblasComplex*     A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           int*                ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrfStridedBatched(hipblasHandle_t       handle,
                                                           const int             n,
                                                           hipblasDoubleComplex* A,
                                                           const int             lda,
                                                           const hipblasStride   strideA,
                                                           int*                  ipiv,
                                                           const hipblasStride   strideP,
                                                           int*                  info,
                                                           const int             batch_count);

// getrs
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrs(hipblasHandle_t          handle,
                                             const hipblasOperation_t trans,
                                             const int                n,
                                             const int                nrhs,
                                             float*                   A,
                                             const int                lda,
                                             const int*               ipiv,
                                             float*                   B,
                                             const int                ldb,
                                             int*                     info);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrs(hipblasHandle_t          handle,
                                             const hipblasOperation_t trans,
                                             const int                n,
                                             const int                nrhs,
                                             double*                  A,
                                             const int                lda,
                                             const int*               ipiv,
                                             double*                  B,
                                             const int                ldb,
                                             int*                     info);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrs(hipblasHandle_t          handle,
                                             const hipblasOperation_t trans,
                                             const int                n,
                                             const int                nrhs,
                                             hipblasComplex*          A,
                                             const int                lda,
                                             const int*               ipiv,
                                             hipblasComplex*          B,
                                             const int                ldb,
                                             int*                     info);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrs(hipblasHandle_t          handle,
                                             const hipblasOperation_t trans,
                                             const int                n,
                                             const int                nrhs,
                                             hipblasDoubleComplex*    A,
                                             const int                lda,
                                             const int*               ipiv,
                                             hipblasDoubleComplex*    B,
                                             const int                ldb,
                                             int*                     info);

// getrs_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrsBatched(hipblasHandle_t          handle,
                                                    const hipblasOperation_t trans,
                                                    const int                n,
                                                    const int                nrhs,
                                                    float* const             A[],
                                                    const int                lda,
                                                    const int*               ipiv,
                                                    float* const             B[],
                                                    const int                ldb,
                                                    int*                     info,
                                                    const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t          handle,
                                                    const hipblasOperation_t trans,
                                                    const int                n,
                                                    const int                nrhs,
                                                    double* const            A[],
                                                    const int                lda,
                                                    const int*               ipiv,
                                                    double* const            B[],
                                                    const int                ldb,
                                                    int*                     info,
                                                    const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrsBatched(hipblasHandle_t          handle,
                                                    const hipblasOperation_t trans,
                                                    const int                n,
                                                    const int                nrhs,
                                                    hipblasComplex* const    A[],
                                                    const int                lda,
                                                    const int*               ipiv,
                                                    hipblasComplex* const    B[],
                                                    const int                ldb,
                                                    int*                     info,
                                                    const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrsBatched(hipblasHandle_t             handle,
                                                    const hipblasOperation_t    trans,
                                                    const int                   n,
                                                    const int                   nrhs,
                                                    hipblasDoubleComplex* const A[],
                                                    const int                   lda,
                                                    const int*                  ipiv,
                                                    hipblasDoubleComplex* const B[],
                                                    const int                   ldb,
                                                    int*                        info,
                                                    const int                   batch_count);

// getrs_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrsStridedBatched(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           float*                   A,
                                                           const int                lda,
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           float*                   B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrsStridedBatched(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           double*                  A,
                                                           const int                lda,
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           double*                  B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrsStridedBatched(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           hipblasComplex*          A,
                                                           const int                lda,
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           hipblasComplex*          B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrsStridedBatched(hipblasHandle_t          handle,
                                                           const hipblasOperation_t trans,
                                                           const int                n,
                                                           const int                nrhs,
                                                           hipblasDoubleComplex*    A,
                                                           const int                lda,
                                                           const hipblasStride      strideA,
                                                           const int*               ipiv,
                                                           const hipblasStride      strideP,
                                                           hipblasDoubleComplex*    B,
                                                           const int                ldb,
                                                           const hipblasStride      strideB,
                                                           int*                     info,
                                                           const int                batch_count);

// getri_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgetriBatched(hipblasHandle_t handle,
                                                    const int       n,
                                                    float* const    A[],
                                                    const int       lda,
                                                    int*            ipiv,
                                                    float* const    C[],
                                                    const int       ldc,
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle,
                                                    const int       n,
                                                    double* const   A[],
                                                    const int       lda,
                                                    int*            ipiv,
                                                    double* const   C[],
                                                    const int       ldc,
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgetriBatched(hipblasHandle_t       handle,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    int*                  ipiv,
                                                    hipblasComplex* const C[],
                                                    const int             ldc,
                                                    int*                  info,
                                                    const int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t             handle,
                                                    const int                   n,
                                                    hipblasDoubleComplex* const A[],
                                                    const int                   lda,
                                                    int*                        ipiv,
                                                    hipblasDoubleComplex* const C[],
                                                    const int                   ldc,
                                                    int*                        info,
                                                    const int                   batch_count);

// geqrf
HIPBLAS_EXPORT hipblasStatus_t hipblasSgeqrf(hipblasHandle_t handle,
                                             const int       m,
                                             const int       n,
                                             float*          A,
                                             const int       lda,
                                             float*          ipiv,
                                             int*            info);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,
                                             const int       m,
                                             const int       n,
                                             double*         A,
                                             const int       lda,
                                             double*         ipiv,
                                             int*            info);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,
                                             const int       m,
                                             const int       n,
                                             hipblasComplex* A,
                                             const int       lda,
                                             hipblasComplex* ipiv,
                                             int*            info);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeqrf(hipblasHandle_t       handle,
                                             const int             m,
                                             const int             n,
                                             hipblasDoubleComplex* A,
                                             const int             lda,
                                             hipblasDoubleComplex* ipiv,
                                             int*                  info);

// geqrf_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle,
                                                    const int       m,
                                                    const int       n,
                                                    float* const    A[],
                                                    const int       lda,
                                                    float* const    ipiv[],
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle,
                                                    const int       m,
                                                    const int       n,
                                                    double* const   A[],
                                                    const int       lda,
                                                    double* const   ipiv[],
                                                    int*            info,
                                                    const int       batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeqrfBatched(hipblasHandle_t       handle,
                                                    const int             m,
                                                    const int             n,
                                                    hipblasComplex* const A[],
                                                    const int             lda,
                                                    hipblasComplex* const ipiv[],
                                                    int*                  info,
                                                    const int             batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeqrfBatched(hipblasHandle_t             handle,
                                                    const int                   m,
                                                    const int                   n,
                                                    hipblasDoubleComplex* const A[],
                                                    const int                   lda,
                                                    hipblasDoubleComplex* const ipiv[],
                                                    int*                        info,
                                                    const int                   batch_count);

// geqrf_strided_batched
HIPBLAS_EXPORT hipblasStatus_t hipblasSgeqrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           m,
                                                           const int           n,
                                                           float*              A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           float*              ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasDgeqrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           m,
                                                           const int           n,
                                                           double*             A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           double*             ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasCgeqrfStridedBatched(hipblasHandle_t     handle,
                                                           const int           m,
                                                           const int           n,
                                                           hipblasComplex*     A,
                                                           const int           lda,
                                                           const hipblasStride strideA,
                                                           hipblasComplex*     ipiv,
                                                           const hipblasStride strideP,
                                                           int*                info,
                                                           const int           batch_count);

HIPBLAS_EXPORT hipblasStatus_t hipblasZgeqrfStridedBatched(hipblasHandle_t       handle,
                                                           const int             m,
                                                           const int             n,
                                                           hipblasDoubleComplex* A,
                                                           const int             lda,
                                                           const hipblasStride   strideA,
                                                           hipblasDoubleComplex* ipiv,
                                                           const hipblasStride   strideP,
                                                           int*                  info,
                                                           const int             batch_count);

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

HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t    handle,
                                                    hipblasOperation_t trans_a,
                                                    hipblasOperation_t trans_b,
                                                    int                m,
                                                    int                n,
                                                    int                k,
                                                    const void*        alpha,
                                                    const void*        a[],
                                                    hipblasDatatype_t  a_type,
                                                    int                lda,
                                                    const void*        b[],
                                                    hipblasDatatype_t  b_type,
                                                    int                ldb,
                                                    const void*        beta,
                                                    void*              c[],
                                                    hipblasDatatype_t  c_type,
                                                    int                ldc,
                                                    int                batch_count,
                                                    hipblasDatatype_t  compute_type,
                                                    hipblasGemmAlgo_t  algo);

HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t    handle,
                                                           hipblasOperation_t trans_a,
                                                           hipblasOperation_t trans_b,
                                                           int                m,
                                                           int                n,
                                                           int                k,
                                                           const void*        alpha,
                                                           const void*        a,
                                                           hipblasDatatype_t  a_type,
                                                           int                lda,
                                                           hipblasStride      stride_A,
                                                           const void*        b,
                                                           hipblasDatatype_t  b_type,
                                                           int                ldb,
                                                           hipblasStride      stride_B,
                                                           const void*        beta,
                                                           void*              c,
                                                           hipblasDatatype_t  c_type,
                                                           int                ldc,
                                                           hipblasStride      stride_C,
                                                           int                batch_count,
                                                           hipblasDatatype_t  compute_type,
                                                           hipblasGemmAlgo_t  algo);

// trsm_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasTrsmEx(hipblasHandle_t    handle,
                                             hipblasSideMode_t  side,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int                m,
                                             int                n,
                                             const void*        alpha,
                                             void*              A,
                                             int                lda,
                                             void*              B,
                                             int                ldb,
                                             const void*        invA,
                                             int                invA_size,
                                             hipblasDatatype_t  compute_type);

HIPBLAS_EXPORT hipblasStatus_t hipblasTrsmBatchedEx(hipblasHandle_t    handle,
                                                    hipblasSideMode_t  side,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int                m,
                                                    int                n,
                                                    const void*        alpha,
                                                    void*              A,
                                                    int                lda,
                                                    void*              B,
                                                    int                ldb,
                                                    int                batch_count,
                                                    const void*        invA,
                                                    int                invA_size,
                                                    hipblasDatatype_t  compute_type);

HIPBLAS_EXPORT hipblasStatus_t hipblasTrsmStridedBatchedEx(hipblasHandle_t    handle,
                                                           hipblasSideMode_t  side,
                                                           hipblasFillMode_t  uplo,
                                                           hipblasOperation_t transA,
                                                           hipblasDiagType_t  diag,
                                                           int                m,
                                                           int                n,
                                                           const void*        alpha,
                                                           void*              A,
                                                           int                lda,
                                                           hipblasStride      stride_A,
                                                           void*              B,
                                                           int                ldb,
                                                           hipblasStride      stride_B,
                                                           int                batch_count,
                                                           const void*        invA,
                                                           int                invA_size,
                                                           hipblasStride      stride_invA,
                                                           hipblasDatatype_t  compute_type);

// // syrk_ex
// HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkEx(hipblasHandle_t       handle,
//                                               hipblasFillMode_t     uplo,
//                                               hipblasOperation_t    trans,
//                                               int                   n,
//                                               int                   k,
//                                               const hipblasComplex* alpha,
//                                               const void*           A,
//                                               hipblasDatatype_t     Atype,
//                                               int                   lda,
//                                               const hipblasComplex* beta,
//                                               hipblasComplex*       C,
//                                               hipblasDatatype_t     Ctype,
//                                               int                   ldc);

// // herk_ex
// HIPBLAS_EXPORT hipblasStatus_t hipblasCherkEx(hipblasHandle_t    handle,
//                                               hipblasFillMode_t  uplo,
//                                               hipblasOperation_t trans,
//                                               int                n,
//                                               int                k,
//                                               const float*       alpha,
//                                               const void*        A,
//                                               hipblasDatatype_t  Atype,
//                                               int                lda,
//                                               const float*       beta,
//                                               hipblasComplex*    C,
//                                               hipblasDatatype_t  Ctype,
//                                               int                ldc);

// axpy_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyEx(hipblasHandle_t   handle,
                                             int               n,
                                             const void*       alpha,
                                             hipblasDatatype_t alphaType,
                                             const void*       x,
                                             hipblasDatatype_t xType,
                                             int               incx,
                                             void*             y,
                                             hipblasDatatype_t yType,
                                             int               incy,
                                             hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyBatchedEx(hipblasHandle_t   handle,
                                                    int               n,
                                                    const void*       alpha,
                                                    hipblasDatatype_t alphaType,
                                                    const void*       x,
                                                    hipblasDatatype_t xType,
                                                    int               incx,
                                                    void*             y,
                                                    hipblasDatatype_t yType,
                                                    int               incy,
                                                    int               batch_count,
                                                    hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyStridedBatchedEx(hipblasHandle_t   handle,
                                                           int               n,
                                                           const void*       alpha,
                                                           hipblasDatatype_t alphaType,
                                                           const void*       x,
                                                           hipblasDatatype_t xType,
                                                           int               incx,
                                                           hipblasStride     stridex,
                                                           void*             y,
                                                           hipblasDatatype_t yType,
                                                           int               incy,
                                                           hipblasStride     stridey,
                                                           int               batch_count,
                                                           hipblasDatatype_t executionType);

// dot_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasDotEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            const void*       y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasDotcEx(hipblasHandle_t   handle,
                                             int               n,
                                             const void*       x,
                                             hipblasDatatype_t xType,
                                             int               incx,
                                             const void*       y,
                                             hipblasDatatype_t yType,
                                             int               incy,
                                             void*             result,
                                             hipblasDatatype_t resultType,
                                             hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasDotBatchedEx(hipblasHandle_t   handle,
                                                   int               n,
                                                   const void*       x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   const void*       y,
                                                   hipblasDatatype_t yType,
                                                   int               incy,
                                                   int               batch_count,
                                                   void*             result,
                                                   hipblasDatatype_t resultType,
                                                   hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasDotcBatchedEx(hipblasHandle_t   handle,
                                                    int               n,
                                                    const void*       x,
                                                    hipblasDatatype_t xType,
                                                    int               incx,
                                                    const void*       y,
                                                    hipblasDatatype_t yType,
                                                    int               incy,
                                                    int               batch_count,
                                                    void*             result,
                                                    hipblasDatatype_t resultType,
                                                    hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasDotStridedBatchedEx(hipblasHandle_t   handle,
                                                          int               n,
                                                          const void*       x,
                                                          hipblasDatatype_t xType,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          const void*       y,
                                                          hipblasDatatype_t yType,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          int               batch_count,
                                                          void*             result,
                                                          hipblasDatatype_t resultType,
                                                          hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasDotcStridedBatchedEx(hipblasHandle_t   handle,
                                                           int               n,
                                                           const void*       x,
                                                           hipblasDatatype_t xType,
                                                           int               incx,
                                                           hipblasStride     stridex,
                                                           const void*       y,
                                                           hipblasDatatype_t yType,
                                                           int               incy,
                                                           hipblasStride     stridey,
                                                           int               batch_count,
                                                           void*             result,
                                                           hipblasDatatype_t resultType,
                                                           hipblasDatatype_t executionType);

// nrm2_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2Ex(hipblasHandle_t   handle,
                                             int               n,
                                             const void*       x,
                                             hipblasDatatype_t xType,
                                             int               incx,
                                             void*             result,
                                             hipblasDatatype_t resultType,
                                             hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2BatchedEx(hipblasHandle_t   handle,
                                                    int               n,
                                                    const void*       x,
                                                    hipblasDatatype_t xType,
                                                    int               incx,
                                                    int               batch_count,
                                                    void*             result,
                                                    hipblasDatatype_t resultType,
                                                    hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2StridedBatchedEx(hipblasHandle_t   handle,
                                                           int               n,
                                                           const void*       x,
                                                           hipblasDatatype_t xType,
                                                           int               incx,
                                                           hipblasStride     stridex,
                                                           int               batch_count,
                                                           void*             result,
                                                           hipblasDatatype_t resultType,
                                                           hipblasDatatype_t executionType);

// rot_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasRotEx(hipblasHandle_t   handle,
                                            int               n,
                                            void*             x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            void*             y,
                                            hipblasDatatype_t yType,
                                            int               incy,
                                            const void*       c,
                                            const void*       s,
                                            hipblasDatatype_t csType,
                                            hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasRotBatchedEx(hipblasHandle_t   handle,
                                                   int               n,
                                                   void*             x,
                                                   hipblasDatatype_t xType,
                                                   int               incx,
                                                   void*             y,
                                                   hipblasDatatype_t yType,
                                                   int               incy,
                                                   const void*       c,
                                                   const void*       s,
                                                   hipblasDatatype_t csType,
                                                   int               batch_count,
                                                   hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasRotStridedBatchedEx(hipblasHandle_t   handle,
                                                          int               n,
                                                          void*             x,
                                                          hipblasDatatype_t xType,
                                                          int               incx,
                                                          hipblasStride     stridex,
                                                          void*             y,
                                                          hipblasDatatype_t yType,
                                                          int               incy,
                                                          hipblasStride     stridey,
                                                          const void*       c,
                                                          const void*       s,
                                                          hipblasDatatype_t csType,
                                                          int               batch_count,
                                                          hipblasDatatype_t executionType);

// scal_ex
HIPBLAS_EXPORT hipblasStatus_t hipblasScalEx(hipblasHandle_t   handle,
                                             int               n,
                                             const void*       alpha,
                                             hipblasDatatype_t alphaType,
                                             void*             x,
                                             hipblasDatatype_t xType,
                                             int               incx,
                                             hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasScalBatchedEx(hipblasHandle_t   handle,
                                                    int               n,
                                                    const void*       alpha,
                                                    hipblasDatatype_t alphaType,
                                                    void*             x,
                                                    hipblasDatatype_t xType,
                                                    int               incx,
                                                    int               batch_count,
                                                    hipblasDatatype_t executionType);

HIPBLAS_EXPORT hipblasStatus_t hipblasScalStridedBatchedEx(hipblasHandle_t   handle,
                                                           int               n,
                                                           const void*       alpha,
                                                           hipblasDatatype_t alphaType,
                                                           void*             x,
                                                           hipblasDatatype_t xType,
                                                           int               incx,
                                                           hipblasStride     stridex,
                                                           int               batch_count,
                                                           hipblasDatatype_t executionType);

/*! HIPBLAS Auxiliary API

    \details
    hipblasStatusToString

    Returns string representing hipblasStatus_t value

    @param[in]
    status  [hipblasStatus_t]
            hipBLAS status to convert to string
*/

HIPBLAS_EXPORT const char* hipblasStatusToString(hipblasStatus_t status);

#ifdef __cplusplus
}
#endif

#endif
