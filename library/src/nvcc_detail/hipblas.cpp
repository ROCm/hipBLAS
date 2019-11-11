/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipblas.h"
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cublasOperation_t hipOperationToCudaOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return CUBLAS_OP_N;

    case HIPBLAS_OP_T:
        return CUBLAS_OP_T;

    case HIPBLAS_OP_C:
        return CUBLAS_OP_C;

    default:
        throw "Non existent OP";
    }
}

hipblasOperation_t CudaOperationToHIPOperation(cublasOperation_t op)
{
    switch(op)
    {
    case CUBLAS_OP_N:
        return HIPBLAS_OP_N;

    case CUBLAS_OP_T:
        return HIPBLAS_OP_T;

    case CUBLAS_OP_C:
        return HIPBLAS_OP_C;

    default:
        throw "Non existent OP";
    }
}

cublasFillMode_t hipFillToCudaFill(hipblasFillMode_t fill)
{
    switch(fill)
    {
    case HIPBLAS_FILL_MODE_UPPER:
        return CUBLAS_FILL_MODE_UPPER;
    case HIPBLAS_FILL_MODE_LOWER:
        return CUBLAS_FILL_MODE_LOWER;
    default:
        throw "Non existent FILL";
    }
}

hipblasFillMode_t CudaFillToHIPFill(cublasFillMode_t fill)
{
    switch(fill)
    {
    case CUBLAS_FILL_MODE_UPPER:
        return HIPBLAS_FILL_MODE_UPPER;
    case CUBLAS_FILL_MODE_LOWER:
        return HIPBLAS_FILL_MODE_LOWER;
    default:
        throw "Non existent FILL";
    }
}

cublasDiagType_t hipDiagonalToCudaDiagonal(hipblasDiagType_t diagonal)
{
    switch(diagonal)
    {
    case HIPBLAS_DIAG_NON_UNIT:
        return CUBLAS_DIAG_NON_UNIT;
    case HIPBLAS_DIAG_UNIT:
        return CUBLAS_DIAG_UNIT;
    default:
        throw "Non existent DIAGONAL";
    }
}

hipblasDiagType_t CudaDiagonalToHIPDiagonal(cublasDiagType_t diagonal)
{
    switch(diagonal)
    {
    case CUBLAS_DIAG_NON_UNIT:
        return HIPBLAS_DIAG_NON_UNIT;
    case CUBLAS_DIAG_UNIT:
        return HIPBLAS_DIAG_UNIT;
    default:
        throw "Non existent DIAGONAL";
    }
}

cublasSideMode_t hipSideToCudaSide(hipblasSideMode_t side)
{
    switch(side)
    {
    case HIPBLAS_SIDE_LEFT:
        return CUBLAS_SIDE_LEFT;
    case HIPBLAS_SIDE_RIGHT:
        return CUBLAS_SIDE_RIGHT;
    default:
        throw "Non existent SIDE";
    }
}

hipblasSideMode_t CudaSideToHIPSide(cublasSideMode_t side)
{
    switch(side)
    {
    case CUBLAS_SIDE_LEFT:
        return HIPBLAS_SIDE_LEFT;
    case CUBLAS_SIDE_RIGHT:
        return HIPBLAS_SIDE_RIGHT;
    default:
        throw "Non existent SIDE";
    }
}

cublasPointerMode_t HIPPointerModeToCudaPointerMode(hipblasPointerMode_t mode)
{
    switch(mode)
    {
    case HIPBLAS_POINTER_MODE_HOST:
        return CUBLAS_POINTER_MODE_HOST;

    case HIPBLAS_POINTER_MODE_DEVICE:
        return CUBLAS_POINTER_MODE_DEVICE;

    default:
        throw "Non existent PointerMode";
    }
}

hipblasPointerMode_t CudaPointerModeToHIPPointerMode(cublasPointerMode_t mode)
{
    switch(mode)
    {
    case CUBLAS_POINTER_MODE_HOST:
        return HIPBLAS_POINTER_MODE_HOST;

    case CUBLAS_POINTER_MODE_DEVICE:
        return HIPBLAS_POINTER_MODE_DEVICE;

    default:
        throw "Non existent PointerMode";
    }
}

hipblasStatus_t hipCUBLASStatusToHIPStatus(cublasStatus_t cuStatus)
{
    switch(cuStatus)
    {
    case CUBLAS_STATUS_SUCCESS:
        return HIPBLAS_STATUS_SUCCESS;
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case CUBLAS_STATUS_ALLOC_FAILED:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case CUBLAS_STATUS_INVALID_VALUE:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case CUBLAS_STATUS_MAPPING_ERROR:
        return HIPBLAS_STATUS_MAPPING_ERROR;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return HIPBLAS_STATUS_EXECUTION_FAILED;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    default:
        throw "Unimplemented status";
    }
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId)
{
    return hipCUBLASStatusToHIPStatus(cublasSetStream((cublasHandle_t)handle, streamId));
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId)
{
    return hipCUBLASStatusToHIPStatus(cublasGetStream((cublasHandle_t)handle, streamId));
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
{
    return hipCUBLASStatusToHIPStatus(cublasCreate((cublasHandle_t*)handle));
}

// TODO broke common API semantics, think about this again.
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
{
    return hipCUBLASStatusToHIPStatus(cublasDestroy((cublasHandle_t)handle));
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSetPointerMode((cublasHandle_t)handle, HIPPointerModeToCudaPointerMode(mode)));
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
{
    cublasPointerMode_t cublasMode;
    cublasStatus        status = cublasGetPointerMode((cublasHandle_t)handle, &cublasMode);
    *mode                      = CudaPointerModeToHIPPointerMode(cublasMode);
    return hipCUBLASStatusToHIPStatus(status);
}

// note: no handle
hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSetVector(n, elemSize, x, incx, y, incy)); // HGSOS no need for handle
}

// note: no handle
hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasGetVector(n, elemSize, x, incx, y, incy)); // HGSOS no need for handle
}

// note: no handle
hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    return hipCUBLASStatusToHIPStatus(cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}

// note: no handle
hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    return hipCUBLASStatusToHIPStatus(cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t hipblasSgeam(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasSgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  beta,
                                                  B,
                                                  ldb,
                                                  C,
                                                  ldc));
}

hipblasStatus_t hipblasDgeam(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasDgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  beta,
                                                  B,
                                                  ldb,
                                                  C,
                                                  ldc));
}

// MAX
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(cublasIsamax((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(cublasIdamax((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamax((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}

hipblasStatus_t
    hipblasIzamax(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamax((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}

// MIN
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(cublasIsamin((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(cublasIdamin((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamin((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}

hipblasStatus_t
    hipblasIzamin(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamin((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}

// ASUM
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return hipCUBLASStatusToHIPStatus(cublasSasum((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return hipCUBLASStatusToHIPStatus(cublasDasum((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasScasum(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasScasum((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}

hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDzasum((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}

// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(cublasSasum((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t hipblasDasumBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumBatched(
    hipblasHandle_t handle, int n, const hipComplex* const x[], int incx, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumBatched(
    hipblasHandle_t handle, int n, const hipDoubleComplex* const x[], int incx, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(
    hipblasHandle_t handle, int n, const float* x, int incx, int stridex, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDasumStridedBatched(
    hipblasHandle_t handle, int n, const double* x, int incx, int stridex, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumStridedBatched(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, int stridex, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumStridedBatched(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int stridex, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// axpy
hipblasStatus_t hipblasSaxpy(
    hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,
                             int             n,
                             const double*   alpha,
                             const double*   x,
                             int             incx,
                             double*         y,
                             int             incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* alpha,
                             const hipComplex* x,
                             int               incx,
                             hipComplex*       y,
                             int               incy)
{
    return hipCUBLASStatusToHIPStatus(cublasCaxpy(
        (cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx, (cuComplex*)y, incy));
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* alpha,
                             const hipDoubleComplex* x,
                             int                     incx,
                             hipDoubleComplex*       y,
                             int                     incy)
{
    return hipCUBLASStatusToHIPStatus(cublasZaxpy((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle,
                                    int             n,
                                    const float*    alpha,
                                    const float*    x,
                                    int             incx,
                                    float*          y,
                                    int             incy,
                                    int             batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return hipCUBLASStatusToHIPStatus(
        cublasSaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t handle,
                                    int             n,
                                    const double*   alpha,
                                    const double*   x,
                                    int             incx,
                                    double*         y,
                                    int             incy,
                                    int             batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return hipCUBLASStatusToHIPStatus(
        cublasDaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}

// copy
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(cublasScopy((cublasHandle_t)handle, n, x, incx, y, incy));
}

hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(cublasDcopy((cublasHandle_t)handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasCcopy(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasCcopy((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}

hipblasStatus_t hipblasZcopy(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             hipDoubleComplex*       y,
                             int                     incy)
{
    return hipCUBLASStatusToHIPStatus(cublasZcopy(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}

// copy_batched
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float* const x[], int incx, float* const y[],
    int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double* const x[], int incx, double* const y[],
    int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t handle, int n, const hipComplex* const x[], int incx, hipComplex* const y[],
    int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t handle, int n, const hipDoubleComplex* const x[], int incx, hipDoubleComplex* const y[],
    int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// copy_strided_batched
hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle, int n, const float* x, int incx, int stridex, float* y,
    int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle, int n, const double* x, int incx, int stridex, double* y,
    int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int stridex, hipComplex* y,
    int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int stridex, hipDoubleComplex* y,
    int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dot
hipblasStatus_t hipblasSdot(hipblasHandle_t handle,
                            int             n,
                            const float*    x,
                            int             incx,
                            const float*    y,
                            int             incy,
                            float*          result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle,
                            int             n,
                            const double*   x,
                            int             incx,
                            const double*   y,
                            int             incy,
                            double*         result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* x,
                             int               incx,
                             const hipComplex* y,
                             int               incy,
                             hipComplex*       result)
{
    return hipCUBLASStatusToHIPStatus(cublasCdotc(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* x,
                             int               incx,
                             const hipComplex* y,
                             int               incy,
                             hipComplex*       result)
{
    return hipCUBLASStatusToHIPStatus(cublasCdotu(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             const hipDoubleComplex* y,
                             int                     incy,
                             hipDoubleComplex*       result)
{
    return hipCUBLASStatusToHIPStatus(cublasZdotc((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             const hipDoubleComplex* y,
                             int                     incy,
                             hipDoubleComplex*       result)
{
    return hipCUBLASStatusToHIPStatus(cublasZdotu((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}

// dot_batched
hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batchCount,
                                   float*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(
    //     cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasDdotBatched(hipblasHandle_t     handle,
                                   int                 n,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   int                 batchCount,
                                   double*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t        handle,
                                   int                     n,
                                   const hipComplex* const x[],
                                   int                     incx,
                                   const hipComplex* const y[],
                                   int                     incy,
                                   int                     batchCount,
                                   hipComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t        handle,
                                   int                     n,
                                   const hipComplex* const x[],
                                   int                     incx,
                                   const hipComplex* const y[],
                                   int                     incy,
                                   int                     batchCount,
                                   hipComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t              handle,
                                   int                           n,
                                   const hipDoubleComplex* const x[],
                                   int                           incx,
                                   const hipDoubleComplex* const y[],
                                   int                           incy,
                                   int                           batchCount,
                                   hipDoubleComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t              handle,
                                   int                           n,
                                   const hipDoubleComplex* const x[],
                                   int                           incx,
                                   const hipDoubleComplex* const y[],
                                   int                           incy,
                                   int                           batchCount,
                                   hipDoubleComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dot_strided_batched
hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          int             stridex,
                                          const float*    y,
                                          int             incy,
                                          int             stridey,
                                          int             batchCount,
                                          float*          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          int             stridex,
                                          const double*   y,
                                          int             incy,
                                          int             stridey,
                                          int             batchCount,
                                          double*         result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* x,
                                           int               incx,
                                           int               stridex,
                                           const hipComplex* y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount,
                                           hipComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* x,
                                           int               incx,
                                           int               stridex,
                                           const hipComplex* y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount,
                                           hipComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           const hipDoubleComplex* y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batchCount,
                                           hipDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           const hipDoubleComplex* y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batchCount,
                                           hipDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2
hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return hipCUBLASStatusToHIPStatus(cublasSnrm2((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return hipCUBLASStatusToHIPStatus(cublasDnrm2((cublasHandle_t)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasScnrm2((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}

hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDznrm2((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}

// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t handle, int n, const hipComplex* const x[], int incx, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t handle, int n, const hipDoubleComplex* const x[], int incx, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle, int n, const float* x, int incx, int stridex, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle, int n, const double* x, int incx, int stridex, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int stridex, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int stridex, int batchCount, double* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rot
// hipblasStatus_t hipblasSrot(hipblasHandle_t handle,
//                             int             n,
//                             float*          x,
//                             int             incx,
//                             float*          y,
//                             int             incy,
//                             const float*    c,
//                             const float*    s)
// {
//     return hipCUBLASStatusToHIPStatus(cublasSrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
// }

// hipblasStatus_t hipblasDrot(hipblasHandle_t handle,
//                             int             n,
//                             double*         x,
//                             int             incx,
//                             double*         y,
//                             int             incy,
//                             const double*   c,
//                             const double*   s)
// {
//     return hipCUBLASStatusToHIPStatus(cublasDrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
// }

// scal
hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(cublasSscal((cublasHandle_t)handle, n, alpha, x, incx));
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(cublasDscal((cublasHandle_t)handle, n, alpha, x, incx));
}

hipblasStatus_t
    hipblasCscal(hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(
        cublasCscal((cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx));
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(
        cublasCsscal((cublasHandle_t)handle, n, alpha, (cuComplex*)x, incx));
}

hipblasStatus_t hipblasZscal(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(
        cublasZscal((cublasHandle_t)handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx));
}

hipblasStatus_t
    hipblasZdscal(hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx)
{
    return hipCUBLASStatusToHIPStatus(
        cublasZdscal((cublasHandle_t)handle, n, alpha, (cuDoubleComplex*)x, incx));
}

// scal_batched
hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(cublasSscal((cublasHandle_t)handle, n, alpha, x, incx));
}
hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalBatched(
    hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalBatched(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, hipComplex* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx, int stridex, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// swap
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(cublasSswap((cublasHandle_t)handle, n, x, incx, y, incy));
}

hipblasStatus_t
    hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(cublasDswap((cublasHandle_t)handle, n, x, incx, y, incy));
}

hipblasStatus_t
    hipblasCswap(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(
        cublasCswap((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}

hipblasStatus_t hipblasZswap(
    hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
{
    return hipCUBLASStatusToHIPStatus(cublasZswap(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}

// swap_batched
hipblasStatus_t hipblasSswapBatched(hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapBatched(hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle, int n, hipComplex* x[], int incx, hipComplex* y[], int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t handle, int n, hipDoubleComplex* x[], int incx, hipDoubleComplex* y[], int incy, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// swap_strided_batched
hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle, int n, float* x, int incx, int stridex, float* y, int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle, int n, double* x, int incx, int stridex, double* y, int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle, int n, hipComplex* x, int incx, int stridex, hipComplex* y, int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, int stridex, hipDoubleComplex* y, int incy, int stridey, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gemv
hipblasStatus_t hipblasSgemv(hipblasHandle_t    handle,
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
                             int                incy)
{
    return hipCUBLASStatusToHIPStatus(cublasSgemv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t    handle,
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
                             int                incy)
{
    return hipCUBLASStatusToHIPStatus(cublasDgemv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}

hipblasStatus_t hipblasCgemv(hipblasHandle_t    handle,
                             hipblasOperation_t trans,
                             int                m,
                             int                n,
                             const hipComplex*  alpha,
                             const hipComplex*  A,
                             int                lda,
                             const hipComplex*  x,
                             int                incx,
                             const hipComplex*  beta,
                             hipComplex*        y,
                             int                incy)
{
    return hipCUBLASStatusToHIPStatus(cublasCgemv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZgemv(hipblasHandle_t         handle,
                             hipblasOperation_t      trans,
                             int                     m,
                             int                     n,
                             const hipDoubleComplex* alpha,
                             const hipDoubleComplex* A,
                             int                     lda,
                             const hipDoubleComplex* x,
                             int                     incx,
                             const hipDoubleComplex* beta,
                             hipDoubleComplex*       y,
                             int                     incy)
{
    return hipCUBLASStatusToHIPStatus(cublasZgemv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}

// gemv_batched
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // return hipCUBLASStatusToHIPStatus(cublasSgemv((cublasHandle_t)handle,
    //                                               hipOperationToCudaOperation(trans),
    //                                               m,
    //                                               n,
    //                                               alpha,
    //                                               A,
    //                                               lda,
    //                                               x,
    //                                               incx,
    //                                               beta,
    //                                               y,
    //                                               incy));
}

hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // return hipCUBLASStatusToHIPStatus(cublasDgemv((cublasHandle_t)handle,
    //                                               hipOperationToCudaOperation(trans),
    //                                               m,
    //                                               n,
    //                                               alpha,
    //                                               A,
    //                                               lda,
    //                                               x,
    //                                               incx,
    //                                               beta,
    //                                               y,
    //                                               incy));
}

hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t         handle,
                                    hipblasOperation_t      trans,
                                    int                     m,
                                    int                     n,
                                    const hipComplex*       alpha,
                                    const hipComplex* const A[],
                                    int                     lda,
                                    const hipComplex* const x[],
                                    int                     incx,
                                    const hipComplex*       beta,
                                    hipComplex* const       y[],
                                    int                     incy,
                                    int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t               handle,
                                    hipblasOperation_t            trans,
                                    int                           m,
                                    int                           n,
                                    const hipDoubleComplex*       alpha,
                                    const hipDoubleComplex* const A[],
                                    int                           lda,
                                    const hipDoubleComplex* const x[],
                                    int                           incx,
                                    const hipDoubleComplex*       beta,
                                    hipDoubleComplex* const       y[],
                                    int                           incy,
                                    int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gemv_strided_batched
hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // return hipCUBLASStatusToHIPStatus(cublasSgemv((cublasHandle_t)handle,
    //                                               hipOperationToCudaOperation(trans),
    //                                               m,
    //                                               n,
    //                                               alpha,
    //                                               A,
    //                                               lda,
    //                                               x,
    //                                               incx,
    //                                               beta,
    //                                               y,
    //                                               incy));
}

hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
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
                                           int                incy,
                                           int                stridey,
                                           int                batchCount)
{
    // TODO warn user that function was demoted to ignore batch
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // return hipCUBLASStatusToHIPStatus(cublasDgemv((cublasHandle_t)handle,
    //                                               hipOperationToCudaOperation(trans),
    //                                               m,
    //                                               n,
    //                                               alpha,
    //                                               A,
    //                                               lda,
    //                                               x,
    //                                               incx,
    //                                               beta,
    //                                               y,
    //                                               incy));
}

hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           const hipComplex*  alpha,
                                           const hipComplex*  A,
                                           int                lda,
                                           int                strideA,
                                           const hipComplex*  x,
                                           int                incx,
                                           int                stridex,
                                           const hipComplex*  beta,
                                           hipComplex*        y,
                                           int                incy,
                                           int                stridey,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t         handle,
                                           hipblasOperation_t      trans,
                                           int                     m,
                                           int                     n,
                                           const hipDoubleComplex* alpha,
                                           const hipDoubleComplex* A,
                                           int                     lda,
                                           int                     strideA,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           const hipDoubleComplex* beta,
                                           hipDoubleComplex*       y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// ger
hipblasStatus_t hipblasSger(hipblasHandle_t handle,
                            int             m,
                            int             n,
                            const float*    alpha,
                            const float*    x,
                            int             incx,
                            const float*    y,
                            int             incy,
                            float*          A,
                            int             lda)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t hipblasDger(hipblasHandle_t handle,
                            int             m,
                            int             n,
                            const double*   alpha,
                            const double*   x,
                            int             incx,
                            const double*   y,
                            int             incy,
                            double*         A,
                            int             lda)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

// ger_batched
hipblasStatus_t hipblasSgerBatched(hipblasHandle_t    handle,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgerBatched(hipblasHandle_t     handle,
                                   int                 m,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// ger_strided_batched
hipblasStatus_t hipblasSgerStridedBatched(hipblasHandle_t handle,
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
                                          int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t  handle,
                                          int              m,
                                          int              n,
                                          const double*    alpha,
                                          const double*    x,
                                          int              incx,
                                          int              stridex,
                                          const double*    y,
                                          int              incy,
                                          int              stridey,
                                          double*          A,
                                          int              lda,
                                          int              strideA,
                                          int              batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr
hipblasStatus_t hipblasSsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            A,
                            int               lda)
{
    return hipCUBLASStatusToHIPStatus(
        cublasSsyr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, A, lda));
}

hipblasStatus_t hipblasDsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           A,
                            int               lda)
{
    return hipCUBLASStatusToHIPStatus(
        cublasDsyr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, A, lda));
}

// syr_batched
hipblasStatus_t hipblasSsyrBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       A[],
                                   int                lda,
                                   int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       A[],
                                   int                 lda,
                                   int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr_strided_batched
hipblasStatus_t hipblasSsyrStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          float*            A,
                                          int               lda,
                                          int               strideA,
                                          int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrStridedBatched(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const double*     alpha,
                                         const double*     x,
                                         int               incx,
                                         int               stridex,
                                         double*           A,
                                         int               lda,
                                         int               strideA,
                                         int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trsv
hipblasStatus_t hipblasStrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
{
    return hipCUBLASStatusToHIPStatus(cublasStrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}

hipblasStatus_t hipblasDtrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
{
    return hipCUBLASStatusToHIPStatus(cublasDtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}

// trsm
hipblasStatus_t hipblasStrsm(hipblasHandle_t    handle,
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
                             int                ldb)
{
    return hipCUBLASStatusToHIPStatus(cublasStrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb));
}

hipblasStatus_t hipblasDtrsm(hipblasHandle_t    handle,
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
                             int                ldb)
{
    return hipCUBLASStatusToHIPStatus(cublasDtrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb));
}

// gemm
hipblasStatus_t hipblasHgemm(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasHgemm((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (__half*)alpha,
                                                  (__half*)A,
                                                  lda,
                                                  (__half*)B,
                                                  ldb,
                                                  (__half*)beta,
                                                  (__half*)C,
                                                  ldc));
}

hipblasStatus_t hipblasSgemm(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasSgemm((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc));
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t    handle,
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
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasDgemm((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc));
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const hipComplex*  alpha,
                             const hipComplex*  A,
                             int                lda,
                             const hipComplex*  B,
                             int                ldb,
                             const hipComplex*  beta,
                             hipComplex*        C,
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasCgemm((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)C,
                                                  ldc));
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t    handle,
                             hipblasOperation_t transa,
                             hipblasOperation_t transb,
                             int                m,
                             int                n,
                             int                k,
                             const hipComplex*  alpha,
                             const hipComplex*  A,
                             int                lda,
                             const hipComplex*  B,
                             int                ldb,
                             const hipComplex*  beta,
                             hipComplex*        C,
                             int                ldc)
{
    return hipCUBLASStatusToHIPStatus(cublasZgemm((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}

/*   complex not supported in hcc
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t
transb,
                           int m, int n, int k,  const hipComplex *alpha, hipComplex *A, int lda,
hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
  return hipCUBLASStatusToHIPStatus(cublasCgemm((cublasHandle_t) handle,
hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,
lda, B,  ldb, beta, C,  ldc));
}
*/



/*  complex not supported in hcc
hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa,
hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A[], int
lda, const hipComplex *B[], int ldb, const hipComplex *beta, hipComplex *C[], int ldc, int
batchCount){

  return hipCUBLASStatusToHIPStatus(cublasCgemmBatched((cublasHandle_t) handle,
hipOperationToCudaOperation(transa),  hipOperationToCudaOperation(transb), m,  n,  k, alpha, A,
lda, B,  ldb, beta, C,  ldc, batchCount));
}
*/

// gemm_batched
hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return hipCUBLASStatusToHIPStatus(cublasSgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         A,
                                                         lda,
                                                         B,
                                                         ldb,
                                                         beta,
                                                         C,
                                                         ldc,
                                                         batchCount));
}

hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return hipCUBLASStatusToHIPStatus(cublasDgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         A,
                                                         lda,
                                                         B,
                                                         ldb,
                                                         beta,
                                                         C,
                                                         ldc,
                                                         batchCount));
}

// gemm_strided_batched
hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return hipCUBLASStatusToHIPStatus(cublasSgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                alpha,
                                                                const_cast<float*>(A),
                                                                lda,
                                                                bsa,
                                                                const_cast<float*>(B),
                                                                ldb,
                                                                bsb,
                                                                beta,
                                                                C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}

hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return hipCUBLASStatusToHIPStatus(cublasDgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                alpha,
                                                                const_cast<double*>(A),
                                                                lda,
                                                                bsa,
                                                                const_cast<double*>(B),
                                                                ldb,
                                                                bsb,
                                                                beta,
                                                                C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}


#ifdef __cplusplus
}
#endif
