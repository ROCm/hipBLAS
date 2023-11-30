/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "hipblas.h"
#include "exceptions.hpp"
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
    case HIPBLAS_FILL_MODE_FULL:
        return CUBLAS_FILL_MODE_FULL;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
    case CUBLAS_FILL_MODE_FULL:
        return HIPBLAS_FILL_MODE_FULL;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

cudaDataType_t HIPDatatypeToCudaDatatype(hipblasDatatype_t type)
{
    switch(type)
    {
    case HIPBLAS_R_16F:
        return CUDA_R_16F;

    case HIPBLAS_R_32F:
        return CUDA_R_32F;

    case HIPBLAS_R_64F:
        return CUDA_R_64F;

    case HIPBLAS_C_16F:
        return CUDA_C_16F;

    case HIPBLAS_C_32F:
        return CUDA_C_32F;

    case HIPBLAS_C_64F:
        return CUDA_C_64F;

    case HIPBLAS_R_8I:
        return CUDA_R_8I;

    case HIPBLAS_R_8U:
        return CUDA_R_8U;

    case HIPBLAS_R_32I:
        return CUDA_R_32I;

    case HIPBLAS_C_8I:
        return CUDA_C_8I;

    case HIPBLAS_C_8U:
        return CUDA_C_8U;

    case HIPBLAS_C_32I:
        return CUDA_C_32I;

    case HIPBLAS_R_16B:
        return CUDA_R_16BF;

    case HIPBLAS_C_16B:
        return CUDA_C_16BF;

    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

cudaDataType_t HIPDatatypeToCudaDatatype_v2(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return CUDA_R_16F;

    case HIP_R_32F:
        return CUDA_R_32F;

    case HIP_R_64F:
        return CUDA_R_64F;

    case HIP_C_16F:
        return CUDA_C_16F;

    case HIP_C_32F:
        return CUDA_C_32F;

    case HIP_C_64F:
        return CUDA_C_64F;

    case HIP_R_8I:
        return CUDA_R_8I;

    case HIP_R_8U:
        return CUDA_R_8U;

    case HIP_R_32I:
        return CUDA_R_32I;

    case HIP_C_8I:
        return CUDA_C_8I;

    case HIP_C_8U:
        return CUDA_C_8U;

    case HIP_C_32I:
        return CUDA_C_32I;

    case HIP_R_16BF:
        return CUDA_R_16BF;

    case HIP_C_16BF:
        return CUDA_C_16BF;

    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

cublasComputeType_t HIPComputetypeToCudaComputetype(hipblasComputeType_t type)
{
    switch(type)
    {
    case HIPBLAS_COMPUTE_16F:
        return CUBLAS_COMPUTE_16F;

    case HIPBLAS_COMPUTE_16F_PEDANTIC:
        return CUBLAS_COMPUTE_16F_PEDANTIC;

    case HIPBLAS_COMPUTE_32F:
        return CUBLAS_COMPUTE_32F;

    case HIPBLAS_COMPUTE_32F_PEDANTIC:
        return CUBLAS_COMPUTE_32F_PEDANTIC;

    case HIPBLAS_COMPUTE_32F_FAST_16F:
        return CUBLAS_COMPUTE_32F_FAST_16F;

    case HIPBLAS_COMPUTE_32F_FAST_16BF:
        return CUBLAS_COMPUTE_32F_FAST_16BF;

    case HIPBLAS_COMPUTE_32F_FAST_TF32:
        return CUBLAS_COMPUTE_32F_FAST_TF32;

    case HIPBLAS_COMPUTE_64F:
        return CUBLAS_COMPUTE_64F;

    case HIPBLAS_COMPUTE_64F_PEDANTIC:
        return CUBLAS_COMPUTE_64F_PEDANTIC;

    case HIPBLAS_COMPUTE_32I:
        return CUBLAS_COMPUTE_32I;

    case HIPBLAS_COMPUTE_32I_PEDANTIC:
        return CUBLAS_COMPUTE_32I_PEDANTIC;

    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

cublasGemmAlgo_t HIPGemmAlgoToCudaGemmAlgo(hipblasGemmAlgo_t algo)
{
    // Only support Default Algo for now
    switch(algo)
    {
    case HIPBLAS_GEMM_DEFAULT:
        return CUBLAS_GEMM_DEFAULT;

    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

cublasAtomicsMode_t HIPAtomicsModeToCudaAtomicsMode(hipblasAtomicsMode_t mode)

{
    switch(mode)
    {
    case HIPBLAS_ATOMICS_NOT_ALLOWED:
        return CUBLAS_ATOMICS_NOT_ALLOWED;
    case HIPBLAS_ATOMICS_ALLOWED:
        return CUBLAS_ATOMICS_ALLOWED;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

hipblasAtomicsMode_t CudaAtomicsModeToHIPAtomicsMode(cublasAtomicsMode_t mode)
{
    switch(mode)
    {
    case CUBLAS_ATOMICS_NOT_ALLOWED:
        return HIPBLAS_ATOMICS_NOT_ALLOWED;
    case CUBLAS_ATOMICS_ALLOWED:
        return HIPBLAS_ATOMICS_ALLOWED;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
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
        return HIPBLAS_STATUS_UNKNOWN;
    }
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSetStream((cublasHandle_t)handle, streamId));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGetStream((cublasHandle_t)handle, streamId));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCreate((cublasHandle_t*)handle));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// TODO broke common API semantics, think about this again.
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDestroy((cublasHandle_t)handle));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSetPointerMode((cublasHandle_t)handle, HIPPointerModeToCudaPointerMode(mode)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
try
{
    cublasPointerMode_t cublasMode;
    cublasStatus_t      status = cublasGetPointerMode((cublasHandle_t)handle, &cublasMode);
    *mode                      = CudaPointerModeToHIPPointerMode(cublasMode);
    return hipCUBLASStatusToHIPStatus(status);
}
catch(...)
{
    return exception_to_hipblas_status();
}

// note: no handle
hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSetVector(n, elemSize, x, incx, y, incy)); // HGSOS no need for handle
}
catch(...)
{
    return exception_to_hipblas_status();
}

// note: no handle
hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGetVector(n, elemSize, x, incx, y, incy)); // HGSOS no need for handle
}
catch(...)
{
    return exception_to_hipblas_status();
}

// note: no handle
hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// note: no handle
hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSetVectorAsync(n, elemSize, x, incx, y, incy, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGetVectorAsync(n, elemSize, x, incx, y, incy, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// atomics mode
hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSetAtomicsMode(
        (cublasHandle_t)handle, HIPAtomicsModeToCudaAtomicsMode(atomics_mode)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGetAtomicsMode((cublasHandle_t)handle, (cublasAtomicsMode_t*)atomics_mode));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasIsamax((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasIdamax((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamax((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamax((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamax_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamax((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax_v2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamax((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax_batched
hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxBatched_v2(hipblasHandle_t         handle,
                                        int                     n,
                                        const hipComplex* const x[],
                                        int                     incx,
                                        int                     batchCount,
                                        int*                    result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxBatched_v2(hipblasHandle_t               handle,
                                        int                           n,
                                        const hipDoubleComplex* const x[],
                                        int                           incx,
                                        int                           batchCount,
                                        int*                          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// amax_strided_batched
hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcamaxStridedBatched_v2(hipblasHandle_t   handle,
                                               int               n,
                                               const hipComplex* x,
                                               int               incx,
                                               hipblasStride     stridex,
                                               int               batchCount,
                                               int*              result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzamaxStridedBatched_v2(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipDoubleComplex* x,
                                               int                     incx,
                                               hipblasStride           stridex,
                                               int                     batchCount,
                                               int*                    result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// amin
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasIsamin((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasIdamin((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamin((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamin(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamin((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamin_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIcamin((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamin_v2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasIzamin((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amin_batched
hipblasStatus_t hipblasIsaminBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdaminBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batchCount, int* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     int*                              result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminBatched_v2(hipblasHandle_t         handle,
                                        int                     n,
                                        const hipComplex* const x[],
                                        int                     incx,
                                        int                     batchCount,
                                        int*                    result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminBatched_v2(hipblasHandle_t               handle,
                                        int                           n,
                                        const hipDoubleComplex* const x[],
                                        int                           incx,
                                        int                           batchCount,
                                        int*                          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// amin_strided_batched
hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount,
                                            int*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            int*                  result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            int*                        result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIcaminStridedBatched_v2(hipblasHandle_t   handle,
                                               int               n,
                                               const hipComplex* x,
                                               int               incx,
                                               hipblasStride     stridex,
                                               int               batchCount,
                                               int*              result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasIzaminStridedBatched_v2(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipDoubleComplex* x,
                                               int                     incx,
                                               hipblasStride           stridex,
                                               int                     batchCount,
                                               int*                    result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// ASUM
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSasum((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDasum((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasScasum((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDzasum((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScasum_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasScasum((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasum_v2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDzasum((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(cublasSasum((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumBatched_v2(hipblasHandle_t         handle,
                                        int                     n,
                                        const hipComplex* const x[],
                                        int                     incx,
                                        int                     batchCount,
                                        float*                  result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumBatched_v2(hipblasHandle_t               handle,
                                        int                           n,
                                        const hipDoubleComplex* const x[],
                                        int                           incx,
                                        int                           batchCount,
                                        double*                       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScasumStridedBatched_v2(hipblasHandle_t   handle,
                                               int               n,
                                               const hipComplex* x,
                                               int               incx,
                                               hipblasStride     stridex,
                                               int               batchCount,
                                               float*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDzasumStridedBatched_v2(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipDoubleComplex* x,
                                               int                     incx,
                                               hipblasStride           stridex,
                                               int                     batchCount,
                                               double*                 result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// axpy
hipblasStatus_t hipblasHaxpy(hipblasHandle_t    handle,
                             int                n,
                             const hipblasHalf* alpha,
                             const hipblasHalf* x,
                             int                incx,
                             hipblasHalf*       y,
                             int                incy)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSaxpy(
    hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,
                             int             n,
                             const double*   alpha,
                             const double*   x,
                             int             incx,
                             double*         y,
                             int             incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             hipblasComplex*       y,
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCaxpy(
        (cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZaxpy((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpy_v2(hipblasHandle_t   handle,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                hipComplex*       y,
                                int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCaxpy(
        (cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpy_v2(hipblasHandle_t         handle,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZaxpy((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// axpy_batched
hipblasStatus_t hipblasHaxpyBatched(hipblasHandle_t          handle,
                                    int                      n,
                                    const hipblasHalf*       alpha,
                                    const hipblasHalf* const x[],
                                    int                      incx,
                                    hipblasHalf* const       y[],
                                    int                      incy,
                                    int                      batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(
    //     cublasSaxpy((cublasHandle_t)handle, n, alpha, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyBatched_v2(hipblasHandle_t         handle,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       hipComplex* const       y[],
                                       int                     incy,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyBatched_v2(hipblasHandle_t               handle,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       hipDoubleComplex* const       y[],
                                       int                           incy,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// axpy_strided_batched
hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t    handle,
                                           int                n,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           hipblasHalf*       y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCaxpyStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZaxpyStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// copy
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasScopy((cublasHandle_t)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDcopy((cublasHandle_t)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopy(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCcopy((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZcopy(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZcopy(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopy_v2(
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCcopy((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZcopy_v2(hipblasHandle_t         handle,
                                int                     n,
                                const hipDoubleComplex* x,
                                int                     incx,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZcopy(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// copy_batched
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyBatched_v2(hipblasHandle_t         handle,
                                       int                     n,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       hipComplex* const       y[],
                                       int                     incy,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyBatched_v2(hipblasHandle_t               handle,
                                       int                           n,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       hipDoubleComplex* const       y[],
                                       int                           incy,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// copy_strided_batched
hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCcopyStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZcopyStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dot
hipblasStatus_t hipblasHdot(hipblasHandle_t    handle,
                            int                n,
                            const hipblasHalf* x,
                            int                incx,
                            const hipblasHalf* y,
                            int                incy,
                            hipblasHalf*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasBfdot(hipblasHandle_t        handle,
                             int                    n,
                             const hipblasBfloat16* x,
                             int                    incx,
                             const hipblasBfloat16* y,
                             int                    incy,
                             hipblasBfloat16*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdot(hipblasHandle_t handle,
                            int             n,
                            const float*    x,
                            int             incx,
                            const float*    y,
                            int             incy,
                            float*          result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle,
                            int             n,
                            const double*   x,
                            int             incx,
                            const double*   y,
                            int             incy,
                            double*         result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdotc(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdotu(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdotc((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdotu((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotc_v2(hipblasHandle_t   handle,
                                int               n,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdotc(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotu_v2(hipblasHandle_t   handle,
                                int               n,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdotu(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotc_v2(hipblasHandle_t         handle,
                                int                     n,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdotc((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotu_v2(hipblasHandle_t         handle,
                                int                     n,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdotu((cublasHandle_t)handle,
                                                  n,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// dot_batched
hipblasStatus_t hipblasHdotBatched(hipblasHandle_t          handle,
                                   int                      n,
                                   const hipblasHalf* const x[],
                                   int                      incx,
                                   const hipblasHalf* const y[],
                                   int                      incy,
                                   int                      batchCount,
                                   hipblasHalf*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(
    //     cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t              handle,
                                    int                          n,
                                    const hipblasBfloat16* const x[],
                                    int                          incx,
                                    const hipblasBfloat16* const y[],
                                    int                          incy,
                                    int                          batchCount,
                                    hipblasBfloat16*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batchCount,
                                   float*             result)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(
    //     cublasSdot((cublasHandle_t)handle, n, x, incx, y, incy, result));
}
catch(...)
{
    return exception_to_hipblas_status();
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

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batchCount,
                                    hipblasComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batchCount,
                                    hipblasDoubleComplex*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcBatched_v2(hipblasHandle_t         handle,
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

hipblasStatus_t hipblasCdotuBatched_v2(hipblasHandle_t         handle,
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

hipblasStatus_t hipblasZdotcBatched_v2(hipblasHandle_t               handle,
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

hipblasStatus_t hipblasZdotuBatched_v2(hipblasHandle_t               handle,
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
hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t    handle,
                                          int                n,
                                          const hipblasHalf* x,
                                          int                incx,
                                          hipblasStride      stridex,
                                          const hipblasHalf* y,
                                          int                incy,
                                          hipblasStride      stridey,
                                          int                batchCount,
                                          hipblasHalf*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                           int                    n,
                                           const hipblasBfloat16* x,
                                           int                    incx,
                                           hipblasStride          stridex,
                                           const hipblasBfloat16* y,
                                           int                    incy,
                                           hipblasStride          stridey,
                                           int                    batchCount,
                                           hipblasBfloat16*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          float*          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batchCount,
                                          double*         result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount,
                                           hipblasComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batchCount,
                                           hipblasDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotcStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount,
                                              hipComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdotuStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount,
                                              hipComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotcStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount,
                                              hipDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdotuStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount,
                                              hipDoubleComplex*       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2
hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSnrm2((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDnrm2((cublasHandle_t)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasScnrm2((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDznrm2((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScnrm2_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasScnrm2((cublasHandle_t)handle, n, (cuComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2_v2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDznrm2((cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2Batched_v2(hipblasHandle_t         handle,
                                        int                     n,
                                        const hipComplex* const x[],
                                        int                     incx,
                                        int                     batchCount,
                                        float*                  result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2Batched_v2(hipblasHandle_t               handle,
                                        int                           n,
                                        const hipDoubleComplex* const x[],
                                        int                           incx,
                                        int                           batchCount,
                                        double*                       result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScnrm2StridedBatched_v2(hipblasHandle_t   handle,
                                               int               n,
                                               const hipComplex* x,
                                               int               incx,
                                               hipblasStride     stridex,
                                               int               batchCount,
                                               float*            result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDznrm2StridedBatched_v2(hipblasHandle_t         handle,
                                               int                     n,
                                               const hipDoubleComplex* x,
                                               int                     incx,
                                               hipblasStride           stridex,
                                               int                     batchCount,
                                               double*                 result)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rot
hipblasStatus_t hipblasSrot(hipblasHandle_t handle,
                            int             n,
                            float*          x,
                            int             incx,
                            float*          y,
                            int             incy,
                            const float*    c,
                            const float*    s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrot(hipblasHandle_t handle,
                            int             n,
                            double*         x,
                            int             incx,
                            double*         y,
                            int             incy,
                            const double*   c,
                            const double*   s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDrot((cublasHandle_t)handle, n, x, incx, y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrot(hipblasHandle_t       handle,
                            int                   n,
                            hipblasComplex*       x,
                            int                   incx,
                            hipblasComplex*       y,
                            int                   incy,
                            const float*          c,
                            const hipblasComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCrot(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, c, (cuComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,
                             int             n,
                             hipblasComplex* x,
                             int             incx,
                             hipblasComplex* y,
                             int             incy,
                             const float*    c,
                             const float*    s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCsrot((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrot(hipblasHandle_t             handle,
                            int                         n,
                            hipblasDoubleComplex*       x,
                            int                         incx,
                            hipblasDoubleComplex*       y,
                            int                         incy,
                            const double*               c,
                            const hipblasDoubleComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZrot((cublasHandle_t)handle,
                                                 n,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)y,
                                                 incy,
                                                 c,
                                                 (cuDoubleComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdrot(hipblasHandle_t       handle,
                             int                   n,
                             hipblasDoubleComplex* x,
                             int                   incx,
                             hipblasDoubleComplex* y,
                             int                   incy,
                             const double*         c,
                             const double*         s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdrot(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrot_v2(hipblasHandle_t   handle,
                               int               n,
                               hipComplex*       x,
                               int               incx,
                               hipComplex*       y,
                               int               incy,
                               const float*      c,
                               const hipComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCrot(
        (cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, c, (cuComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsrot_v2(hipblasHandle_t handle,
                                int             n,
                                hipComplex*     x,
                                int             incx,
                                hipComplex*     y,
                                int             incy,
                                const float*    c,
                                const float*    s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCsrot((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrot_v2(hipblasHandle_t         handle,
                               int                     n,
                               hipDoubleComplex*       x,
                               int                     incx,
                               hipDoubleComplex*       y,
                               int                     incy,
                               const double*           c,
                               const hipDoubleComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZrot((cublasHandle_t)handle,
                                                 n,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)y,
                                                 incy,
                                                 c,
                                                 (cuDoubleComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdrot_v2(hipblasHandle_t   handle,
                                int               n,
                                hipDoubleComplex* x,
                                int               incx,
                                hipDoubleComplex* y,
                                int               incy,
                                const double*     c,
                                const double*     s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdrot(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rot_batched
hipblasStatus_t hipblasSrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   float* const    x[],
                                   int             incx,
                                   float* const    y[],
                                   int             incy,
                                   const float*    c,
                                   const float*    s,
                                   int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotBatched(hipblasHandle_t handle,
                                   int             n,
                                   double* const   x[],
                                   int             incx,
                                   double* const   y[],
                                   int             incy,
                                   const double*   c,
                                   const double*   s,
                                   int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotBatched(hipblasHandle_t       handle,
                                   int                   n,
                                   hipblasComplex* const x[],
                                   int                   incx,
                                   hipblasComplex* const y[],
                                   int                   incy,
                                   const float*          c,
                                   const hipblasComplex* s,
                                   int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    hipblasComplex* const y[],
                                    int                   incy,
                                    const float*          c,
                                    const float*          s,
                                    int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotBatched(hipblasHandle_t             handle,
                                   int                         n,
                                   hipblasDoubleComplex* const x[],
                                   int                         incx,
                                   hipblasDoubleComplex* const y[],
                                   int                         incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s,
                                   int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    hipblasDoubleComplex* const y[],
                                    int                         incy,
                                    const double*               c,
                                    const double*               s,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotBatched_v2(hipblasHandle_t   handle,
                                      int               n,
                                      hipComplex* const x[],
                                      int               incx,
                                      hipComplex* const y[],
                                      int               incy,
                                      const float*      c,
                                      const hipComplex* s,
                                      int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotBatched_v2(hipblasHandle_t   handle,
                                       int               n,
                                       hipComplex* const x[],
                                       int               incx,
                                       hipComplex* const y[],
                                       int               incy,
                                       const float*      c,
                                       const float*      s,
                                       int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotBatched_v2(hipblasHandle_t         handle,
                                      int                     n,
                                      hipDoubleComplex* const x[],
                                      int                     incx,
                                      hipDoubleComplex* const y[],
                                      int                     incy,
                                      const double*           c,
                                      const hipDoubleComplex* s,
                                      int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotBatched_v2(hipblasHandle_t         handle,
                                       int                     n,
                                       hipDoubleComplex* const x[],
                                       int                     incx,
                                       hipDoubleComplex* const y[],
                                       int                     incy,
                                       const double*           c,
                                       const double*           s,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rot_strided_batched
hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          float*          x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          float*          y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const float*    c,
                                          const float*    s,
                                          int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          double*         x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          double*         y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          const double*   c,
                                          const double*   s,
                                          int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                          int                   n,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       y,
                                          int                   incy,
                                          hipblasStride         stridey,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    c,
                                           const float*    s,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                          int                         n,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       y,
                                          int                         incy,
                                          hipblasStride               stridey,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           const double*         c,
                                           const double*         s,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotStridedBatched_v2(hipblasHandle_t   handle,
                                             int               n,
                                             hipComplex*       x,
                                             int               incx,
                                             hipblasStride     stridex,
                                             hipComplex*       y,
                                             int               incy,
                                             hipblasStride     stridey,
                                             const float*      c,
                                             const hipComplex* s,
                                             int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsrotStridedBatched_v2(hipblasHandle_t handle,
                                              int             n,
                                              hipComplex*     x,
                                              int             incx,
                                              hipblasStride   stridex,
                                              hipComplex*     y,
                                              int             incy,
                                              hipblasStride   stridey,
                                              const float*    c,
                                              const float*    s,
                                              int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotStridedBatched_v2(hipblasHandle_t         handle,
                                             int                     n,
                                             hipDoubleComplex*       x,
                                             int                     incx,
                                             hipblasStride           stridex,
                                             hipDoubleComplex*       y,
                                             int                     incy,
                                             hipblasStride           stridey,
                                             const double*           c,
                                             const hipDoubleComplex* s,
                                             int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdrotStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              hipDoubleComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              hipDoubleComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              const double*     c,
                                              const double*     s,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotg
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSrotg((cublasHandle_t)handle, a, b, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDrotg((cublasHandle_t)handle, a, b, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotg(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCrotg((cublasHandle_t)handle, (cuComplex*)a, (cuComplex*)b, c, (cuComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotg(hipblasHandle_t       handle,
                             hipblasDoubleComplex* a,
                             hipblasDoubleComplex* b,
                             double*               c,
                             hipblasDoubleComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZrotg(
        (cublasHandle_t)handle, (cuDoubleComplex*)a, (cuDoubleComplex*)b, c, (cuDoubleComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCrotg_v2(hipblasHandle_t handle, hipComplex* a, hipComplex* b, float* c, hipComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCrotg((cublasHandle_t)handle, (cuComplex*)a, (cuComplex*)b, c, (cuComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotg_v2(hipblasHandle_t   handle,
                                hipDoubleComplex* a,
                                hipDoubleComplex* b,
                                double*           c,
                                hipDoubleComplex* s)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZrotg(
        (cublasHandle_t)handle, (cuDoubleComplex*)a, (cuDoubleComplex*)b, c, (cuDoubleComplex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotg_batchced
hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                    float* const    a[],
                                    float* const    b[],
                                    float* const    c[],
                                    float* const    s[],
                                    int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                    double* const   a[],
                                    double* const   b[],
                                    double* const   c[],
                                    double* const   s[],
                                    int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                    hipblasComplex* const a[],
                                    hipblasComplex* const b[],
                                    float* const          c[],
                                    hipblasComplex* const s[],
                                    int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                    hipblasDoubleComplex* const a[],
                                    hipblasDoubleComplex* const b[],
                                    double* const               c[],
                                    hipblasDoubleComplex* const s[],
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgBatched_v2(hipblasHandle_t   handle,
                                       hipComplex* const a[],
                                       hipComplex* const b[],
                                       float* const      c[],
                                       hipComplex* const s[],
                                       int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgBatched_v2(hipblasHandle_t         handle,
                                       hipDoubleComplex* const a[],
                                       hipDoubleComplex* const b[],
                                       double* const           c[],
                                       hipDoubleComplex* const s[],
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotg_strided_batched
hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle,
                                           float*          a,
                                           hipblasStride   stride_a,
                                           float*          b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           float*          s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                           double*         a,
                                           hipblasStride   stride_a,
                                           double*         b,
                                           hipblasStride   stride_b,
                                           double*         c,
                                           hipblasStride   stride_c,
                                           double*         s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                           hipblasComplex* a,
                                           hipblasStride   stride_a,
                                           hipblasComplex* b,
                                           hipblasStride   stride_b,
                                           float*          c,
                                           hipblasStride   stride_c,
                                           hipblasComplex* s,
                                           hipblasStride   stride_s,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                           hipblasDoubleComplex* a,
                                           hipblasStride         stride_a,
                                           hipblasDoubleComplex* b,
                                           hipblasStride         stride_b,
                                           double*               c,
                                           hipblasStride         stride_c,
                                           hipblasDoubleComplex* s,
                                           hipblasStride         stride_s,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCrotgStridedBatched_v2(hipblasHandle_t handle,
                                              hipComplex*     a,
                                              hipblasStride   stride_a,
                                              hipComplex*     b,
                                              hipblasStride   stride_b,
                                              float*          c,
                                              hipblasStride   stride_c,
                                              hipComplex*     s,
                                              hipblasStride   stride_s,
                                              int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZrotgStridedBatched_v2(hipblasHandle_t   handle,
                                              hipDoubleComplex* a,
                                              hipblasStride     stride_a,
                                              hipDoubleComplex* b,
                                              hipblasStride     stride_b,
                                              double*           c,
                                              hipblasStride     stride_c,
                                              hipDoubleComplex* s,
                                              hipblasStride     stride_s,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotm
hipblasStatus_t hipblasSrotm(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSrotm((cublasHandle_t)handle, n, x, incx, y, incy, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotm(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDrotm((cublasHandle_t)handle, n, x, incx, y, incy, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotm_batched
hipblasStatus_t hipblasSrotmBatched(hipblasHandle_t    handle,
                                    int                n,
                                    float* const       x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    const float* const param[],
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    double* const       x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    const double* const param[],
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotm_strided_batched
hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const float*    param,
                                           hipblasStride   strideParam,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           const double*   param,
                                           hipblasStride   strideParam,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotmg
hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSrotmg((cublasHandle_t)handle, d1, d2, x1, y1, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDrotmg((cublasHandle_t)handle, d1, d2, x1, y1, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotmg_batched
hipblasStatus_t hipblasSrotmgBatched(hipblasHandle_t    handle,
                                     float* const       d1[],
                                     float* const       d2[],
                                     float* const       x1[],
                                     const float* const y1[],
                                     float* const       param[],
                                     int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                     double* const       d1[],
                                     double* const       d2[],
                                     double* const       x1[],
                                     const double* const y1[],
                                     double* const       param[],
                                     int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rotmg_strided_batched
hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle,
                                            float*          d1,
                                            hipblasStride   stride_d1,
                                            float*          d2,
                                            hipblasStride   stride_d2,
                                            float*          x1,
                                            hipblasStride   stride_x1,
                                            const float*    y1,
                                            hipblasStride   stride_y1,
                                            float*          param,
                                            hipblasStride   strideParam,
                                            int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                            double*         d1,
                                            hipblasStride   stride_d1,
                                            double*         d2,
                                            hipblasStride   stride_d2,
                                            double*         x1,
                                            hipblasStride   stride_x1,
                                            const double*   y1,
                                            hipblasStride   stride_y1,
                                            double*         param,
                                            hipblasStride   strideParam,
                                            int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// scal
hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSscal((cublasHandle_t)handle, n, alpha, x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDscal((cublasHandle_t)handle, n, alpha, x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscal(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCscal((cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCsscal((cublasHandle_t)handle, n, alpha, (cuComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscal(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasZscal((cublasHandle_t)handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscal(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasZdscal((cublasHandle_t)handle, n, alpha, (cuDoubleComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCscal_v2(hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCscal((cublasHandle_t)handle, n, (cuComplex*)alpha, (cuComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCsscal_v2(hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCsscal((cublasHandle_t)handle, n, alpha, (cuComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscal_v2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasZscal((cublasHandle_t)handle, n, (cuDoubleComplex*)alpha, (cuDoubleComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscal_v2(
    hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasZdscal((cublasHandle_t)handle, n, alpha, (cuDoubleComplex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// scal_batched
hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount)
try
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
    // TODO warn user that function was demoted to ignore batch
    // return hipCUBLASStatusToHIPStatus(cublasSscal((cublasHandle_t)handle, n, alpha, x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                     int                   n,
                                     const float*          alpha,
                                     hipblasComplex* const x[],
                                     int                   incx,
                                     int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const double*               alpha,
                                     hipblasDoubleComplex* const x[],
                                     int                         incx,
                                     int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalBatched_v2(hipblasHandle_t   handle,
                                       int               n,
                                       const hipComplex* alpha,
                                       hipComplex* const x[],
                                       int               incx,
                                       int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalBatched_v2(hipblasHandle_t         handle,
                                       int                     n,
                                       const hipDoubleComplex* alpha,
                                       hipDoubleComplex* const x[],
                                       int                     incx,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalBatched_v2(hipblasHandle_t   handle,
                                        int               n,
                                        const float*      alpha,
                                        hipComplex* const x[],
                                        int               incx,
                                        int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalBatched_v2(hipblasHandle_t         handle,
                                        int                     n,
                                        const double*           alpha,
                                        hipDoubleComplex* const x[],
                                        int                     incx,
                                        int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            hipblasComplex* x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const double*         alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCscalStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              const hipComplex* alpha,
                                              hipComplex*       x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZscalStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsscalStridedBatched_v2(hipblasHandle_t handle,
                                               int             n,
                                               const float*    alpha,
                                               hipComplex*     x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdscalStridedBatched_v2(hipblasHandle_t   handle,
                                               int               n,
                                               const double*     alpha,
                                               hipDoubleComplex* x,
                                               int               incx,
                                               hipblasStride     stridex,
                                               int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// swap
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSswap((cublasHandle_t)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDswap((cublasHandle_t)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswap(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCswap((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswap(hipblasHandle_t       handle,
                             int                   n,
                             hipblasDoubleComplex* x,
                             int                   incx,
                             hipblasDoubleComplex* y,
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZswap(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCswap_v2(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasCswap((cublasHandle_t)handle, n, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswap_v2(
    hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZswap(
        (cublasHandle_t)handle, n, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// swap_batched
hipblasStatus_t hipblasSswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    float* const    x[],
                                    int             incx,
                                    float* const    y[],
                                    int             incy,
                                    int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    double* const   x[],
                                    int             incx,
                                    double* const   y[],
                                    int             incy,
                                    int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    hipblasComplex* const y[],
                                    int                   incy,
                                    int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    hipblasDoubleComplex* const y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapBatched_v2(hipblasHandle_t   handle,
                                       int               n,
                                       hipComplex* const x[],
                                       int               incx,
                                       hipComplex* const y[],
                                       int               incy,
                                       int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapBatched_v2(hipblasHandle_t         handle,
                                       int                     n,
                                       hipDoubleComplex* const x[],
                                       int                     incx,
                                       hipDoubleComplex* const y[],
                                       int                     incy,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// swap_strided_batched
hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           float*          y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           double*         y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           hipblasStride   stridey,
                                           int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCswapStridedBatched_v2(hipblasHandle_t handle,
                                              int             n,
                                              hipComplex*     x,
                                              int             incx,
                                              hipblasStride   stridex,
                                              hipComplex*     y,
                                              int             incy,
                                              hipblasStride   stridey,
                                              int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZswapStridedBatched_v2(hipblasHandle_t   handle,
                                              int               n,
                                              hipDoubleComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              hipDoubleComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gbmv
hipblasStatus_t hipblasSgbmv(hipblasHandle_t    handle,
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
                             int                incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgbmv(hipblasHandle_t    handle,
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
                             int                incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgbmv(hipblasHandle_t       handle,
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
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgbmv(hipblasHandle_t             handle,
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
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgbmv_v2(hipblasHandle_t    handle,
                                hipblasOperation_t trans,
                                int                m,
                                int                n,
                                int                kl,
                                int                ku,
                                const hipComplex*  alpha,
                                const hipComplex*  A,
                                int                lda,
                                const hipComplex*  x,
                                int                incx,
                                const hipComplex*  beta,
                                hipComplex*        y,
                                int                incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgbmv_v2(hipblasHandle_t         handle,
                                hipblasOperation_t      trans,
                                int                     m,
                                int                     n,
                                int                     kl,
                                int                     ku,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgbmv((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gbmv_batched
hipblasStatus_t hipblasSgbmvBatched(hipblasHandle_t    handle,
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
                                    int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgbmvBatched(hipblasHandle_t     handle,
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
                                    int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvBatched(hipblasHandle_t             handle,
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
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvBatched(hipblasHandle_t                   handle,
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
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasOperation_t      trans,
                                       int                     m,
                                       int                     n,
                                       int                     kl,
                                       int                     ku,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex*       beta,
                                       hipComplex* const       y[],
                                       int                     incy,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasOperation_t            trans,
                                       int                           m,
                                       int                           n,
                                       int                           kl,
                                       int                           ku,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       y[],
                                       int                           incy,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gbmv_strided_batched
hipblasStatus_t hipblasSgbmvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgbmvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasOperation_t trans,
                                              int                m,
                                              int                n,
                                              int                kl,
                                              int                ku,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      stride_a,
                                              const hipComplex*  x,
                                              int                incx,
                                              hipblasStride      stride_x,
                                              const hipComplex*  beta,
                                              hipComplex*        y,
                                              int                incy,
                                              hipblasStride      stride_y,
                                              int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgbmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasOperation_t      trans,
                                              int                     m,
                                              int                     n,
                                              int                     kl,
                                              int                     ku,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           stride_a,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stride_x,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stride_y,
                                              int                     batch_count)
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemv(hipblasHandle_t       handle,
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
                             int                   incy)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemv(hipblasHandle_t             handle,
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
                             int                         incy)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemv_v2(hipblasHandle_t    handle,
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemv_v2(hipblasHandle_t         handle,
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemvBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemvBatched_v2(hipblasHandle_t         handle,
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

hipblasStatus_t hipblasZgemvBatched_v2(hipblasHandle_t               handle,
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
                                           hipblasStride      strideA,
                                           const float*       x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           hipblasStride      stridey,
                                           int                batchCount)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgemvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasOperation_t trans,
                                              int                m,
                                              int                n,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              const hipComplex*  x,
                                              int                incx,
                                              hipblasStride      stridex,
                                              const hipComplex*  beta,
                                              hipComplex*        y,
                                              int                incy,
                                              hipblasStride      stridey,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgemvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasOperation_t      trans,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
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
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDger((cublasHandle_t)handle, m, n, alpha, x, incx, y, incy, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeru(hipblasHandle_t       handle,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeru((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgerc(hipblasHandle_t       handle,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgerc((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeru(hipblasHandle_t             handle,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeru((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgerc(hipblasHandle_t             handle,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgerc((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeru_v2(hipblasHandle_t   handle,
                                int               m,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       A,
                                int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeru((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgerc_v2(hipblasHandle_t   handle,
                                int               m,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       A,
                                int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgerc((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeru_v2(hipblasHandle_t         handle,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       A,
                                int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeru((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgerc_v2(hipblasHandle_t         handle,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       A,
                                int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgerc((cublasHandle_t)handle,
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
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

hipblasStatus_t hipblasCgeruBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercBatched(hipblasHandle_t             handle,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeruBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercBatched(hipblasHandle_t                   handle,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeruBatched_v2(hipblasHandle_t         handle,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex* const y[],
                                       int                     incy,
                                       hipComplex* const       A[],
                                       int                     lda,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercBatched_v2(hipblasHandle_t         handle,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex* const y[],
                                       int                     incy,
                                       hipComplex* const       A[],
                                       int                     lda,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeruBatched_v2(hipblasHandle_t               handle,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex* const y[],
                                       int                           incy,
                                       hipDoubleComplex* const       A[],
                                       int                           lda,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercBatched_v2(hipblasHandle_t               handle,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex* const y[],
                                       int                           incy,
                                       hipDoubleComplex* const       A[],
                                       int                           lda,
                                       int                           batchCount)
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
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          float*          A,
                                          int             lda,
                                          hipblasStride   strideA,
                                          int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
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
                                          int             batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeruStridedBatched_v2(hipblasHandle_t   handle,
                                              int               m,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              hipComplex*       A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgercStridedBatched_v2(hipblasHandle_t   handle,
                                              int               m,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              hipComplex*       A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeruStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              hipDoubleComplex*       A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgercStridedBatched_v2(hipblasHandle_t         handle,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              hipDoubleComplex*       A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hbmv
hipblasStatus_t hipblasChbmv(hipblasHandle_t       handle,
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
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhbmv(hipblasHandle_t             handle,
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
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChbmv_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                int               k,
                                const hipComplex* alpha,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* beta,
                                hipComplex*       y,
                                int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhbmv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hbmv_batched
hipblasStatus_t hipblasChbmvBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhbmvBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChbmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       int                     k,
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

hipblasStatus_t hipblasZhbmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       int                           k,
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

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChbmvStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              int               k,
                                              const hipComplex* alpha,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* beta,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhbmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemv
hipblasStatus_t hipblasChemv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* beta,
                             hipblasComplex*       y,
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChemv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhemv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChemv_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* beta,
                                hipComplex*       y,
                                int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChemv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhemv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

// hemv_batched
hipblasStatus_t hipblasChemvBatched(hipblasHandle_t             handle,
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
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvBatched(hipblasHandle_t                   handle,
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
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChemvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex*       beta,
                                       hipComplex* const       y[],
                                       int                     incy,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       y[],
                                       int                           incy,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChemvStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     stride_a,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stride_x,
                                              const hipComplex* beta,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stride_y,
                                              int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           stride_a,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stride_x,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stride_y,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her
hipblasStatus_t hipblasCher(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCher_v2(hipblasHandle_t   handle,
                               hipblasFillMode_t uplo,
                               int               n,
                               const float*      alpha,
                               const hipComplex* x,
                               int               incx,
                               hipComplex*       A,
                               int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher_v2(hipblasHandle_t         handle,
                               hipblasFillMode_t       uplo,
                               int                     n,
                               const double*           alpha,
                               const hipDoubleComplex* x,
                               int                     incx,
                               hipDoubleComplex*       A,
                               int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her_batched
hipblasStatus_t hipblasCherBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherBatched_v2(hipblasHandle_t         handle,
                                      hipblasFillMode_t       uplo,
                                      int                     n,
                                      const float*            alpha,
                                      const hipComplex* const x[],
                                      int                     incx,
                                      hipComplex* const       A[],
                                      int                     lda,
                                      int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherBatched_v2(hipblasHandle_t               handle,
                                      hipblasFillMode_t             uplo,
                                      int                           n,
                                      const double*                 alpha,
                                      const hipDoubleComplex* const x[],
                                      int                           incx,
                                      hipDoubleComplex* const       A[],
                                      int                           lda,
                                      int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherStridedBatched_v2(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int               n,
                                             const float*      alpha,
                                             const hipComplex* x,
                                             int               incx,
                                             hipblasStride     stridex,
                                             hipComplex*       A,
                                             int               lda,
                                             hipblasStride     strideA,
                                             int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherStridedBatched_v2(hipblasHandle_t         handle,
                                             hipblasFillMode_t       uplo,
                                             int                     n,
                                             const double*           alpha,
                                             const hipDoubleComplex* x,
                                             int                     incx,
                                             hipblasStride           stridex,
                                             hipDoubleComplex*       A,
                                             int                     lda,
                                             hipblasStride           strideA,
                                             int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2
hipblasStatus_t hipblasCher2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCher2_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       A,
                                int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       A,
                                int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her2_batched
hipblasStatus_t hipblasCher2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCher2Batched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex* const y[],
                                       int                     incy,
                                       hipComplex* const       A[],
                                       int                     lda,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2Batched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex* const y[],
                                       int                           incy,
                                       hipDoubleComplex* const       A[],
                                       int                           lda,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCher2StridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              hipComplex*       A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2StridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              hipDoubleComplex*       A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpmv
hipblasStatus_t hipblasChpmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* AP,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* beta,
                             hipblasComplex*       y,
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* AP,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChpmv_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* AP,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* beta,
                                hipComplex*       y,
                                int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpmv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* AP,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpmv_batched
hipblasStatus_t hipblasChpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const AP[],
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex*       beta,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const AP[],
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex*       beta,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChpmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const AP[],
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex*       beta,
                                       hipComplex* const       y[],
                                       int                     incy,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const AP[],
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       y[],
                                       int                           incy,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChpmvStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* AP,
                                              hipblasStride     strideAP,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* beta,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* AP,
                                              hipblasStride           strideAP,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr
hipblasStatus_t hipblasChpr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChpr_v2(hipblasHandle_t   handle,
                               hipblasFillMode_t uplo,
                               int               n,
                               const float*      alpha,
                               const hipComplex* x,
                               int               incx,
                               hipComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr_v2(hipblasHandle_t         handle,
                               hipblasFillMode_t       uplo,
                               int                     n,
                               const double*           alpha,
                               const hipDoubleComplex* x,
                               int                     incx,
                               hipDoubleComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr_batched
hipblasStatus_t hipblasChprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const float*                alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChprBatched_v2(hipblasHandle_t         handle,
                                      hipblasFillMode_t       uplo,
                                      int                     n,
                                      const float*            alpha,
                                      const hipComplex* const x[],
                                      int                     incx,
                                      hipComplex* const       AP[],
                                      int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprBatched_v2(hipblasHandle_t               handle,
                                      hipblasFillMode_t             uplo,
                                      int                           n,
                                      const double*                 alpha,
                                      const hipDoubleComplex* const x[],
                                      int                           incx,
                                      hipDoubleComplex* const       AP[],
                                      int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChprStridedBatched_v2(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int               n,
                                             const float*      alpha,
                                             const hipComplex* x,
                                             int               incx,
                                             hipblasStride     stridex,
                                             hipComplex*       AP,
                                             hipblasStride     strideAP,
                                             int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhprStridedBatched_v2(hipblasHandle_t         handle,
                                             hipblasFillMode_t       uplo,
                                             int                     n,
                                             const double*           alpha,
                                             const hipDoubleComplex* x,
                                             int                     incx,
                                             hipblasStride           stridex,
                                             hipDoubleComplex*       AP,
                                             hipblasStride           strideAP,
                                             int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr2
hipblasStatus_t hipblasChpr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChpr2_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChpr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhpr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// hpr2_batched
hipblasStatus_t hipblasChpr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const yp[],
                                    int                         incy,
                                    hipblasComplex* const       AP[],
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const yp[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       AP[],
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChpr2Batched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex* const yp[],
                                       int                     incy,
                                       hipComplex* const       AP[],
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2Batched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex* const yp[],
                                       int                           incy,
                                       hipDoubleComplex* const       AP[],
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChpr2StridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              hipComplex*       AP,
                                              hipblasStride     strideAP,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhpr2StridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              hipDoubleComplex*       AP,
                                              hipblasStride           strideAP,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// sbmv
hipblasStatus_t hipblasSsbmv(hipblasHandle_t   handle,
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
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsbmv(hipblasHandle_t   handle,
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
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx,
                                                  beta,
                                                  y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// sbmv_batched
hipblasStatus_t hipblasSsbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    int                k,
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    int                 k,
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// sbmv_strided_batched
hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spmv
hipblasStatus_t hipblasSspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      AP,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSspmv(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, AP, x, incx, beta, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspmv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     AP,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDspmv(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, AP, x, incx, beta, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spmv_batched
hipblasStatus_t hipblasSspmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const AP[],
                                    const float* const x[],
                                    int                incx,
                                    const float*       beta,
                                    float* const       y[],
                                    int                incy,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const AP[],
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spmv_strided_batched
hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr
hipblasStatus_t hipblasSspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            AP)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSspr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           AP)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDspr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCspr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCspr_v2(hipblasHandle_t   handle,
                               hipblasFillMode_t uplo,
                               int               n,
                               const hipComplex* alpha,
                               const hipComplex* x,
                               int               incx,
                               hipComplex*       AP)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZspr_v2(hipblasHandle_t         handle,
                               hipblasFillMode_t       uplo,
                               int                     n,
                               const hipDoubleComplex* alpha,
                               const hipDoubleComplex* x,
                               int                     incx,
                               hipDoubleComplex*       AP)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr_batched
hipblasStatus_t hipblasSsprBatched(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   int                n,
                                   const float*       alpha,
                                   const float* const x[],
                                   int                incx,
                                   float* const       AP[],
                                   int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsprBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       AP[],
                                   int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprBatched_v2(hipblasHandle_t         handle,
                                      hipblasFillMode_t       uplo,
                                      int                     n,
                                      const hipComplex*       alpha,
                                      const hipComplex* const x[],
                                      int                     incx,
                                      hipComplex* const       AP[],
                                      int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprBatched_v2(hipblasHandle_t               handle,
                                      hipblasFillMode_t             uplo,
                                      int                           n,
                                      const hipDoubleComplex*       alpha,
                                      const hipDoubleComplex* const x[],
                                      int                           incx,
                                      hipDoubleComplex* const       AP[],
                                      int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr_strided_batched
hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          float*            AP,
                                          hipblasStride     strideAP,
                                          int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          hipblasStride     stridex,
                                          double*           AP,
                                          hipblasStride     strideAP,
                                          int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       AP,
                                          hipblasStride         strideAP,
                                          int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       AP,
                                          hipblasStride               strideAP,
                                          int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsprStridedBatched_v2(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int               n,
                                             const hipComplex* alpha,
                                             const hipComplex* x,
                                             int               incx,
                                             hipblasStride     stridex,
                                             hipComplex*       AP,
                                             hipblasStride     strideAP,
                                             int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsprStridedBatched_v2(hipblasHandle_t         handle,
                                             hipblasFillMode_t       uplo,
                                             int                     n,
                                             const hipDoubleComplex* alpha,
                                             const hipDoubleComplex* x,
                                             int                     incx,
                                             hipblasStride           stridex,
                                             hipDoubleComplex*       AP,
                                             hipblasStride           strideAP,
                                             int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr2
hipblasStatus_t hipblasSspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSspr2(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, y, incy, AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           AP)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDspr2(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, y, incy, AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// spr2_batched
hipblasStatus_t hipblasSspr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    const float* const y[],
                                    int                incy,
                                    float* const       AP[],
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    const double* const y[],
                                    int                 incy,
                                    double* const       AP[],
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// spr2_strided_batched
hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symv
hipblasStatus_t hipblasSsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             const float*      beta,
                             float*            y,
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsymv(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, A, lda, x, incx, beta, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymv(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             const double*     beta,
                             double*           y,
                             int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsymv(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, A, lda, x, incx, beta, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* beta,
                             hipblasComplex*       y,
                             int                   incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsymv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       y,
                             int                         incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsymv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymv_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* beta,
                                hipComplex*       y,
                                int               incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsymv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       y,
                                int                     incy)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsymv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

// symv_batched
hipblasStatus_t hipblasSsymvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
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
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymvBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
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

hipblasStatus_t hipblasZsymvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
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

// symv_strided_batched
hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymvStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* beta,
                                              hipComplex*       y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              int                     batchCount)
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
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSsyr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           A,
                            int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDsyr((cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 (cuComplex*)alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 (cuDoubleComplex*)alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr_v2(hipblasHandle_t   handle,
                               hipblasFillMode_t uplo,
                               int               n,
                               const hipComplex* alpha,
                               const hipComplex* x,
                               int               incx,
                               hipComplex*       A,
                               int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 (cuComplex*)alpha,
                                                 (cuComplex*)x,
                                                 incx,
                                                 (cuComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr_v2(hipblasHandle_t         handle,
                               hipblasFillMode_t       uplo,
                               int                     n,
                               const hipDoubleComplex* alpha,
                               const hipDoubleComplex* x,
                               int                     incx,
                               hipDoubleComplex*       A,
                               int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr((cublasHandle_t)handle,
                                                 hipFillToCudaFill(uplo),
                                                 n,
                                                 (cuDoubleComplex*)alpha,
                                                 (cuDoubleComplex*)x,
                                                 incx,
                                                 (cuDoubleComplex*)A,
                                                 lda));
}
catch(...)
{
    return exception_to_hipblas_status();
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

hipblasStatus_t hipblasCsyrBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       A[],
                                   int                         lda,
                                   int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       A[],
                                   int                               lda,
                                   int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrBatched_v2(hipblasHandle_t         handle,
                                      hipblasFillMode_t       uplo,
                                      int                     n,
                                      const hipComplex*       alpha,
                                      const hipComplex* const x[],
                                      int                     incx,
                                      hipComplex* const       A[],
                                      int                     lda,
                                      int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrBatched_v2(hipblasHandle_t               handle,
                                      hipblasFillMode_t             uplo,
                                      int                           n,
                                      const hipDoubleComplex*       alpha,
                                      const hipDoubleComplex* const x[],
                                      int                           incx,
                                      hipDoubleComplex* const       A[],
                                      int                           lda,
                                      int                           batchCount)
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
                                          hipblasStride     stridex,
                                          float*            A,
                                          int               lda,
                                          hipblasStride     strideA,
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
                                          hipblasStride     stridex,
                                          double*           A,
                                          int               lda,
                                          hipblasStride     strideA,
                                          int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          hipblasStride         stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          hipblasStride         strideA,
                                          int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          hipblasStride               stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          hipblasStride               strideA,
                                          int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrStridedBatched_v2(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int               n,
                                             const hipComplex* alpha,
                                             const hipComplex* x,
                                             int               incx,
                                             hipblasStride     stridex,
                                             hipComplex*       A,
                                             int               lda,
                                             hipblasStride     strideA,
                                             int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrStridedBatched_v2(hipblasHandle_t         handle,
                                             hipblasFillMode_t       uplo,
                                             int                     n,
                                             const hipDoubleComplex* alpha,
                                             const hipDoubleComplex* x,
                                             int                     incx,
                                             hipblasStride           stridex,
                                             hipDoubleComplex*       A,
                                             int                     lda,
                                             hipblasStride           strideA,
                                             int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2
hipblasStatus_t hipblasSsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const float*      alpha,
                             const float*      x,
                             int               incx,
                             const float*      y,
                             int               incy,
                             float*            A,
                             int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsyr2(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, y, incy, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2(hipblasHandle_t   handle,
                             hipblasFillMode_t uplo,
                             int               n,
                             const double*     alpha,
                             const double*     x,
                             int               incx,
                             const double*     y,
                             int               incy,
                             double*           A,
                             int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsyr2(
        (cublasHandle_t)handle, hipFillToCudaFill(uplo), n, alpha, x, incx, y, incy, A, lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       A,
                             int                   lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       A,
                             int                         lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2_v2(hipblasHandle_t   handle,
                                hipblasFillMode_t uplo,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* x,
                                int               incx,
                                const hipComplex* y,
                                int               incy,
                                hipComplex*       A,
                                int               lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)y,
                                                  incy,
                                                  (cuComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* x,
                                int                     incx,
                                const hipDoubleComplex* y,
                                int                     incy,
                                hipDoubleComplex*       A,
                                int                     lda)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr2((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)y,
                                                  incy,
                                                  (cuDoubleComplex*)A,
                                                  lda));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2_batched
hipblasStatus_t hipblasSsyr2Batched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
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

hipblasStatus_t hipblasDsyr2Batched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
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

hipblasStatus_t hipblasCsyr2Batched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       A[],
                                    int                         lda,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       A[],
                                    int                               lda,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2Batched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       const hipComplex* const y[],
                                       int                     incy,
                                       hipComplex* const       A[],
                                       int                     lda,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2Batched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       const hipDoubleComplex* const y[],
                                       int                           incy,
                                       hipDoubleComplex* const       A[],
                                       int                           lda,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2_strided_batched
hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2StridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stridex,
                                              const hipComplex* y,
                                              int               incy,
                                              hipblasStride     stridey,
                                              hipComplex*       A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2StridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              const hipDoubleComplex* y,
                                              int                     incy,
                                              hipblasStride           stridey,
                                              hipDoubleComplex*       A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbmv
hipblasStatus_t hipblasStbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                int                k,
                                const hipComplex*  A,
                                int                lda,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* A,
                                int                     lda,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtbmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbmv_batched
hipblasStatus_t hipblasStbmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       int                     k,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       int                           k,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbmvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              int                k,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      stride_a,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stride_x,
                                              int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           stride_a,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stride_x,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbsv
hipblasStatus_t hipblasStbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbsv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                int                k,
                                const hipComplex*  A,
                                int                lda,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbsv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* A,
                                int                     lda,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtbsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbsv_batched
hipblasStatus_t hipblasStbsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       int                     k,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       int                           k,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tbsv_strided_batched
hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtbsvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              int                k,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stridex,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtbsvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpmv
hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  AP,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  AP,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                const hipComplex*  AP,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                const hipDoubleComplex* AP,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtpmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpmv_batched
hipblasStatus_t hipblasStpmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       const hipComplex* const AP[],
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       const hipDoubleComplex* const AP[],
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpmvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              const hipComplex*  AP,
                                              hipblasStride      strideAP,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stridex,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              const hipDoubleComplex* AP,
                                              hipblasStride           strideAP,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpsv
hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  AP,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  AP,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                const hipComplex*  AP,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)AP,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                const hipDoubleComplex* AP,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtpsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)AP,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpsv_batched
hipblasStatus_t hipblasStpsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       const hipComplex* const AP[],
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       const hipDoubleComplex* const AP[],
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtpsvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              const hipComplex*  AP,
                                              hipblasStride      strideAP,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stridex,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtpsvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              const hipDoubleComplex* AP,
                                              hipblasStride           strideAP,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trmv
hipblasStatus_t hipblasStrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                const hipComplex*  A,
                                int                lda,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                const hipDoubleComplex* A,
                                int                     lda,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrmv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmv_batched
hipblasStatus_t hipblasStrmvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stride_x,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stride_x,
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stride_x,
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      stride_a,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stride_x,
                                              int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           stride_a,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stride_x,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trsv
hipblasStatus_t hipblasStrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                n,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   n,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         n,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsv_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                n,
                                const hipComplex*  A,
                                int                lda,
                                hipComplex*        x,
                                int                incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsv_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     n,
                                const hipDoubleComplex* A,
                                int                     lda,
                                hipDoubleComplex*       x,
                                int                     incx)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsv((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsv_batched
hipblasStatus_t hipblasStrsvBatched(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                n,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 n,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         n,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               n,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     n,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       hipComplex* const       x[],
                                       int                     incx,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           n,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       hipDoubleComplex* const       x[],
                                       int                           incx,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsvStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                n,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              hipComplex*        x,
                                              int                incx,
                                              hipblasStride      stridex,
                                              int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsvStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     n,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              hipDoubleComplex*       x,
                                              int                     incx,
                                              hipblasStride           stridex,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

//------------------------------------------------------------------------------------------------------------

// herk
hipblasStatus_t hipblasCherk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const float*          alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const float*          beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCherk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const double*               alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const double*               beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZherk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCherk_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                int                n,
                                int                k,
                                const float*       alpha,
                                const hipComplex*  A,
                                int                lda,
                                const float*       beta,
                                hipComplex*        C,
                                int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCherk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherk_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                int                     n,
                                int                     k,
                                const double*           alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const double*           beta,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZherk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herk_batched
hipblasStatus_t hipblasCherkBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherkBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       int                     n,
                                       int                     k,
                                       const float*            alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const float*            beta,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       int                           n,
                                       int                           k,
                                       const double*                 alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const double*                 beta,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herk_strided_batched
hipblasStatus_t hipblasCherkStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherkStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int                n,
                                              int                k,
                                              const float*       alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              const float*       beta,
                                              hipComplex*        C,
                                              int                ldc,
                                              hipblasStride      strideC,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              int                     n,
                                              int                     k,
                                              const double*           alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const double*           beta,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herkx
hipblasStatus_t hipblasCherkx(hipblasHandle_t       handle,
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
                              int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCherkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuComplex*)alpha,
                                                   (cuComplex*)A,
                                                   lda,
                                                   (cuComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkx(hipblasHandle_t             handle,
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
                              int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZherkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuDoubleComplex*)alpha,
                                                   (cuDoubleComplex*)A,
                                                   lda,
                                                   (cuDoubleComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuDoubleComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCherkx_v2(hipblasHandle_t    handle,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
                                 int                n,
                                 int                k,
                                 const hipComplex*  alpha,
                                 const hipComplex*  A,
                                 int                lda,
                                 const hipComplex*  B,
                                 int                ldb,
                                 const float*       beta,
                                 hipComplex*        C,
                                 int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCherkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuComplex*)alpha,
                                                   (cuComplex*)A,
                                                   lda,
                                                   (cuComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZherkx_v2(hipblasHandle_t         handle,
                                 hipblasFillMode_t       uplo,
                                 hipblasOperation_t      transA,
                                 int                     n,
                                 int                     k,
                                 const hipDoubleComplex* alpha,
                                 const hipDoubleComplex* A,
                                 int                     lda,
                                 const hipDoubleComplex* B,
                                 int                     ldb,
                                 const double*           beta,
                                 hipDoubleComplex*       C,
                                 int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZherkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuDoubleComplex*)alpha,
                                                   (cuDoubleComplex*)A,
                                                   lda,
                                                   (cuDoubleComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuDoubleComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// herkx_batched
hipblasStatus_t hipblasCherkxBatched(hipblasHandle_t             handle,
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
                                     int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxBatched(hipblasHandle_t                   handle,
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
                                     int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherkxBatched_v2(hipblasHandle_t         handle,
                                        hipblasFillMode_t       uplo,
                                        hipblasOperation_t      transA,
                                        int                     n,
                                        int                     k,
                                        const hipComplex*       alpha,
                                        const hipComplex* const A[],
                                        int                     lda,
                                        const hipComplex* const B[],
                                        int                     ldb,
                                        const float*            beta,
                                        hipComplex* const       C[],
                                        int                     ldc,
                                        int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxBatched_v2(hipblasHandle_t               handle,
                                        hipblasFillMode_t             uplo,
                                        hipblasOperation_t            transA,
                                        int                           n,
                                        int                           k,
                                        const hipDoubleComplex*       alpha,
                                        const hipDoubleComplex* const A[],
                                        int                           lda,
                                        const hipDoubleComplex* const B[],
                                        int                           ldb,
                                        const double*                 beta,
                                        hipDoubleComplex* const       C[],
                                        int                           ldc,
                                        int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// herkx_strided_batched
hipblasStatus_t hipblasCherkxStridedBatched(hipblasHandle_t       handle,
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
                                            int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxStridedBatched(hipblasHandle_t             handle,
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
                                            int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCherkxStridedBatched_v2(hipblasHandle_t    handle,
                                               hipblasFillMode_t  uplo,
                                               hipblasOperation_t transA,
                                               int                n,
                                               int                k,
                                               const hipComplex*  alpha,
                                               const hipComplex*  A,
                                               int                lda,
                                               hipblasStride      strideA,
                                               const hipComplex*  B,
                                               int                ldb,
                                               hipblasStride      strideB,
                                               const float*       beta,
                                               hipComplex*        C,
                                               int                ldc,
                                               hipblasStride      strideC,
                                               int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZherkxStridedBatched_v2(hipblasHandle_t         handle,
                                               hipblasFillMode_t       uplo,
                                               hipblasOperation_t      transA,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* alpha,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               hipblasStride           strideA,
                                               const hipDoubleComplex* B,
                                               int                     ldb,
                                               hipblasStride           strideB,
                                               const double*           beta,
                                               hipDoubleComplex*       C,
                                               int                     ldc,
                                               hipblasStride           strideC,
                                               int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2k
hipblasStatus_t hipblasCher2k(hipblasHandle_t       handle,
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
                              int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuComplex*)alpha,
                                                   (cuComplex*)A,
                                                   lda,
                                                   (cuComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2k(hipblasHandle_t             handle,
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
                              int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuDoubleComplex*)alpha,
                                                   (cuDoubleComplex*)A,
                                                   lda,
                                                   (cuDoubleComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuDoubleComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCher2k_v2(hipblasHandle_t    handle,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
                                 int                n,
                                 int                k,
                                 const hipComplex*  alpha,
                                 const hipComplex*  A,
                                 int                lda,
                                 const hipComplex*  B,
                                 int                ldb,
                                 const float*       beta,
                                 hipComplex*        C,
                                 int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCher2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuComplex*)alpha,
                                                   (cuComplex*)A,
                                                   lda,
                                                   (cuComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZher2k_v2(hipblasHandle_t         handle,
                                 hipblasFillMode_t       uplo,
                                 hipblasOperation_t      transA,
                                 int                     n,
                                 int                     k,
                                 const hipDoubleComplex* alpha,
                                 const hipDoubleComplex* A,
                                 int                     lda,
                                 const hipDoubleComplex* B,
                                 int                     ldb,
                                 const double*           beta,
                                 hipDoubleComplex*       C,
                                 int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZher2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
                                                   n,
                                                   k,
                                                   (cuDoubleComplex*)alpha,
                                                   (cuDoubleComplex*)A,
                                                   lda,
                                                   (cuDoubleComplex*)B,
                                                   ldb,
                                                   beta,
                                                   (cuDoubleComplex*)C,
                                                   ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// her2k_batched
hipblasStatus_t hipblasCher2kBatched(hipblasHandle_t             handle,
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
                                     int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kBatched(hipblasHandle_t                   handle,
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
                                     int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCher2kBatched_v2(hipblasHandle_t         handle,
                                        hipblasFillMode_t       uplo,
                                        hipblasOperation_t      transA,
                                        int                     n,
                                        int                     k,
                                        const hipComplex*       alpha,
                                        const hipComplex* const A[],
                                        int                     lda,
                                        const hipComplex* const B[],
                                        int                     ldb,
                                        const float*            beta,
                                        hipComplex* const       C[],
                                        int                     ldc,
                                        int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kBatched_v2(hipblasHandle_t               handle,
                                        hipblasFillMode_t             uplo,
                                        hipblasOperation_t            transA,
                                        int                           n,
                                        int                           k,
                                        const hipDoubleComplex*       alpha,
                                        const hipDoubleComplex* const A[],
                                        int                           lda,
                                        const hipDoubleComplex* const B[],
                                        int                           ldb,
                                        const double*                 beta,
                                        hipDoubleComplex* const       C[],
                                        int                           ldc,
                                        int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// her2k_strided_batched
hipblasStatus_t hipblasCher2kStridedBatched(hipblasHandle_t       handle,
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
                                            int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kStridedBatched(hipblasHandle_t             handle,
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
                                            int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCher2kStridedBatched_v2(hipblasHandle_t    handle,
                                               hipblasFillMode_t  uplo,
                                               hipblasOperation_t transA,
                                               int                n,
                                               int                k,
                                               const hipComplex*  alpha,
                                               const hipComplex*  A,
                                               int                lda,
                                               hipblasStride      strideA,
                                               const hipComplex*  B,
                                               int                ldb,
                                               hipblasStride      strideB,
                                               const float*       beta,
                                               hipComplex*        C,
                                               int                ldc,
                                               hipblasStride      strideC,
                                               int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZher2kStridedBatched_v2(hipblasHandle_t         handle,
                                               hipblasFillMode_t       uplo,
                                               hipblasOperation_t      transA,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* alpha,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               hipblasStride           strideA,
                                               const hipDoubleComplex* B,
                                               int                     ldb,
                                               hipblasStride           strideB,
                                               const double*           beta,
                                               hipDoubleComplex*       C,
                                               int                     ldc,
                                               hipblasStride           strideC,
                                               int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symm
hipblasStatus_t hipblasSsymm(hipblasHandle_t   handle,
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
                             int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsymm(hipblasHandle_t   handle,
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
                             int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  B,
                                                  ldb,
                                                  beta,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymm(hipblasHandle_t       handle,
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
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymm(hipblasHandle_t             handle,
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
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsymm_v2(hipblasHandle_t   handle,
                                hipblasSideMode_t side,
                                hipblasFillMode_t uplo,
                                int               m,
                                int               n,
                                const hipComplex* alpha,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* B,
                                int               ldb,
                                const hipComplex* beta,
                                hipComplex*       C,
                                int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsymm_v2(hipblasHandle_t         handle,
                                hipblasSideMode_t       side,
                                hipblasFillMode_t       uplo,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* B,
                                int                     ldb,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsymm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// symm_batched
hipblasStatus_t hipblasSsymmBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymmBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmBatched_v2(hipblasHandle_t         handle,
                                       hipblasSideMode_t       side,
                                       hipblasFillMode_t       uplo,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const B[],
                                       int                     ldb,
                                       const hipComplex*       beta,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmBatched_v2(hipblasHandle_t               handle,
                                       hipblasSideMode_t             side,
                                       hipblasFillMode_t             uplo,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const B[],
                                       int                           ldb,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// symm_strided_batched
hipblasStatus_t hipblasSsymmStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsymmStridedBatched(hipblasHandle_t   handle,
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
                                           int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsymmStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasSideMode_t side,
                                              hipblasFillMode_t uplo,
                                              int               m,
                                              int               n,
                                              const hipComplex* alpha,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              const hipComplex* B,
                                              int               ldb,
                                              hipblasStride     strideB,
                                              const hipComplex* beta,
                                              hipComplex*       C,
                                              int               ldc,
                                              hipblasStride     strideC,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsymmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasSideMode_t       side,
                                              hipblasFillMode_t       uplo,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* B,
                                              int                     ldb,
                                              hipblasStride           strideB,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syrk
hipblasStatus_t hipblasSsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       beta,
                             float*             C,
                             int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  beta,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrk(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             int                n,
                             int                k,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      beta,
                             double*            C,
                             int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  A,
                                                  lda,
                                                  beta,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrk(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             int                   n,
                             int                   k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* beta,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrk(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             int                         n,
                             int                         k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrk_v2(hipblasHandle_t    handle,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                int                n,
                                int                k,
                                const hipComplex*  alpha,
                                const hipComplex*  A,
                                int                lda,
                                const hipComplex*  beta,
                                hipComplex*        C,
                                int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrk_v2(hipblasHandle_t         handle,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyrk((cublasHandle_t)handle,
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  n,
                                                  k,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// syrk_batched
hipblasStatus_t hipblasSsyrkBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkBatched_v2(hipblasHandle_t         handle,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       int                     n,
                                       int                     k,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex*       beta,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkBatched_v2(hipblasHandle_t               handle,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       int                           n,
                                       int                           k,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syrk_strided_batched
hipblasStatus_t hipblasSsyrkStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int                n,
                                              int                k,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              const hipComplex*  beta,
                                              hipComplex*        C,
                                              int                ldc,
                                              hipblasStride      strideC,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2k
hipblasStatus_t hipblasSsyr2k(hipblasHandle_t    handle,
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
                              int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyr2k(hipblasHandle_t    handle,
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
                              int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2k(hipblasHandle_t       handle,
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
                              int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2k(hipblasHandle_t             handle,
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
                              int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyr2k_v2(hipblasHandle_t    handle,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
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
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyr2k_v2(hipblasHandle_t         handle,
                                 hipblasFillMode_t       uplo,
                                 hipblasOperation_t      transA,
                                 int                     n,
                                 int                     k,
                                 const hipDoubleComplex* alpha,
                                 const hipDoubleComplex* A,
                                 int                     lda,
                                 const hipDoubleComplex* B,
                                 int                     ldb,
                                 const hipDoubleComplex* beta,
                                 hipDoubleComplex*       C,
                                 int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyr2k((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

// syr2k_batched
hipblasStatus_t hipblasSsyr2kBatched(hipblasHandle_t    handle,
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
                                     int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2kBatched(hipblasHandle_t     handle,
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
                                     int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kBatched(hipblasHandle_t             handle,
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
                                     int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kBatched(hipblasHandle_t                   handle,
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
                                     int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kBatched_v2(hipblasHandle_t         handle,
                                        hipblasFillMode_t       uplo,
                                        hipblasOperation_t      transA,
                                        int                     n,
                                        int                     k,
                                        const hipComplex*       alpha,
                                        const hipComplex* const A[],
                                        int                     lda,
                                        const hipComplex* const B[],
                                        int                     ldb,
                                        const hipComplex*       beta,
                                        hipComplex* const       C[],
                                        int                     ldc,
                                        int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kBatched_v2(hipblasHandle_t               handle,
                                        hipblasFillMode_t             uplo,
                                        hipblasOperation_t            transA,
                                        int                           n,
                                        int                           k,
                                        const hipDoubleComplex*       alpha,
                                        const hipDoubleComplex* const A[],
                                        int                           lda,
                                        const hipDoubleComplex* const B[],
                                        int                           ldb,
                                        const hipDoubleComplex*       beta,
                                        hipDoubleComplex* const       C[],
                                        int                           ldc,
                                        int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syr2k_strided_batched
hipblasStatus_t hipblasSsyr2kStridedBatched(hipblasHandle_t    handle,
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
                                            int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyr2kStridedBatched(hipblasHandle_t    handle,
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
                                            int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kStridedBatched(hipblasHandle_t       handle,
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
                                            int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kStridedBatched(hipblasHandle_t             handle,
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
                                            int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyr2kStridedBatched_v2(hipblasHandle_t    handle,
                                               hipblasFillMode_t  uplo,
                                               hipblasOperation_t transA,
                                               int                n,
                                               int                k,
                                               const hipComplex*  alpha,
                                               const hipComplex*  A,
                                               int                lda,
                                               hipblasStride      strideA,
                                               const hipComplex*  B,
                                               int                ldb,
                                               hipblasStride      strideB,
                                               const hipComplex*  beta,
                                               hipComplex*        C,
                                               int                ldc,
                                               hipblasStride      strideC,
                                               int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyr2kStridedBatched_v2(hipblasHandle_t         handle,
                                               hipblasFillMode_t       uplo,
                                               hipblasOperation_t      transA,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* alpha,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               hipblasStride           strideA,
                                               const hipDoubleComplex* B,
                                               int                     ldb,
                                               hipblasStride           strideB,
                                               const hipDoubleComplex* beta,
                                               hipDoubleComplex*       C,
                                               int                     ldc,
                                               hipblasStride           strideC,
                                               int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syrkx
hipblasStatus_t hipblasSsyrkx(hipblasHandle_t    handle,
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
                              int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsyrkx(hipblasHandle_t    handle,
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
                              int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrkx(hipblasHandle_t       handle,
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
                              int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrkx(hipblasHandle_t             handle,
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
                              int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsyrkx_v2(hipblasHandle_t    handle,
                                 hipblasFillMode_t  uplo,
                                 hipblasOperation_t transA,
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
try
{
    return hipCUBLASStatusToHIPStatus(cublasCsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsyrkx_v2(hipblasHandle_t         handle,
                                 hipblasFillMode_t       uplo,
                                 hipblasOperation_t      transA,
                                 int                     n,
                                 int                     k,
                                 const hipDoubleComplex* alpha,
                                 const hipDoubleComplex* A,
                                 int                     lda,
                                 const hipDoubleComplex* B,
                                 int                     ldb,
                                 const hipDoubleComplex* beta,
                                 hipDoubleComplex*       C,
                                 int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZsyrkx((cublasHandle_t)handle,
                                                   hipFillToCudaFill(uplo),
                                                   hipOperationToCudaOperation(transA),
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
catch(...)
{
    return exception_to_hipblas_status();
}

// syrkx_batched
hipblasStatus_t hipblasSsyrkxBatched(hipblasHandle_t    handle,
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
                                     int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkxBatched(hipblasHandle_t     handle,
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
                                     int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkxBatched(hipblasHandle_t             handle,
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
                                     int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkxBatched(hipblasHandle_t                   handle,
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
                                     int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkxBatched_v2(hipblasHandle_t         handle,
                                        hipblasFillMode_t       uplo,
                                        hipblasOperation_t      transA,
                                        int                     n,
                                        int                     k,
                                        const hipComplex*       alpha,
                                        const hipComplex* const A[],
                                        int                     lda,
                                        const hipComplex* const B[],
                                        int                     ldb,
                                        const hipComplex*       beta,
                                        hipComplex* const       C[],
                                        int                     ldc,
                                        int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkxBatched_v2(hipblasHandle_t               handle,
                                        hipblasFillMode_t             uplo,
                                        hipblasOperation_t            transA,
                                        int                           n,
                                        int                           k,
                                        const hipDoubleComplex*       alpha,
                                        const hipDoubleComplex* const A[],
                                        int                           lda,
                                        const hipDoubleComplex* const B[],
                                        int                           ldb,
                                        const hipDoubleComplex*       beta,
                                        hipDoubleComplex* const       C[],
                                        int                           ldc,
                                        int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// syrkx_strided_batched
hipblasStatus_t hipblasSsyrkxStridedBatched(hipblasHandle_t    handle,
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
                                            int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDsyrkxStridedBatched(hipblasHandle_t    handle,
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
                                            int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkxStridedBatched(hipblasHandle_t       handle,
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
                                            int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkxStridedBatched(hipblasHandle_t             handle,
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
                                            int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCsyrkxStridedBatched_v2(hipblasHandle_t    handle,
                                               hipblasFillMode_t  uplo,
                                               hipblasOperation_t transA,
                                               int                n,
                                               int                k,
                                               const hipComplex*  alpha,
                                               const hipComplex*  A,
                                               int                lda,
                                               hipblasStride      strideA,
                                               const hipComplex*  B,
                                               int                ldb,
                                               hipblasStride      strideB,
                                               const hipComplex*  beta,
                                               hipComplex*        C,
                                               int                ldc,
                                               hipblasStride      strideC,
                                               int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZsyrkxStridedBatched_v2(hipblasHandle_t         handle,
                                               hipblasFillMode_t       uplo,
                                               hipblasOperation_t      transA,
                                               int                     n,
                                               int                     k,
                                               const hipDoubleComplex* alpha,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               hipblasStride           strideA,
                                               const hipDoubleComplex* B,
                                               int                     ldb,
                                               hipblasStride           strideB,
                                               const hipDoubleComplex* beta,
                                               hipDoubleComplex*       C,
                                               int                     ldc,
                                               hipblasStride           strideC,
                                               int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// geam
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeam(hipblasHandle_t       handle,
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
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeam(hipblasHandle_t             handle,
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
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeam_v2(hipblasHandle_t    handle,
                                hipblasOperation_t transa,
                                hipblasOperation_t transb,
                                int                m,
                                int                n,
                                const hipComplex*  alpha,
                                const hipComplex*  A,
                                int                lda,
                                const hipComplex*  beta,
                                const hipComplex*  B,
                                int                ldb,
                                hipComplex*        C,
                                int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)beta,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeam_v2(hipblasHandle_t         handle,
                                hipblasOperation_t      transa,
                                hipblasOperation_t      transb,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* beta,
                                const hipDoubleComplex* B,
                                int                     ldb,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeam((cublasHandle_t)handle,
                                                  hipOperationToCudaOperation(transa),
                                                  hipOperationToCudaOperation(transb),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)beta,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// geam_batched
hipblasStatus_t hipblasSgeamBatched(hipblasHandle_t    handle,
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
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgeamBatched(hipblasHandle_t     handle,
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
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeamBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeamBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeamBatched_v2(hipblasHandle_t         handle,
                                       hipblasOperation_t      transa,
                                       hipblasOperation_t      transb,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex*       beta,
                                       const hipComplex* const B[],
                                       int                     ldb,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeamBatched_v2(hipblasHandle_t               handle,
                                       hipblasOperation_t            transa,
                                       hipblasOperation_t            transb,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex*       beta,
                                       const hipDoubleComplex* const B[],
                                       int                           ldb,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// geam_strided_batched
hipblasStatus_t hipblasSgeamStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgeamStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeamStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeamStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeamStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasOperation_t transa,
                                              hipblasOperation_t transb,
                                              int                m,
                                              int                n,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              const hipComplex*  beta,
                                              const hipComplex*  B,
                                              int                ldb,
                                              hipblasStride      strideB,
                                              hipComplex*        C,
                                              int                ldc,
                                              hipblasStride      strideC,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeamStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasOperation_t      transa,
                                              hipblasOperation_t      transb,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* beta,
                                              const hipDoubleComplex* B,
                                              int                     ldb,
                                              hipblasStride           strideB,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemm
hipblasStatus_t hipblasChemm(hipblasHandle_t       handle,
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
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChemm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemm(hipblasHandle_t             handle,
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
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhemm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasChemm_v2(hipblasHandle_t   handle,
                                hipblasSideMode_t side,
                                hipblasFillMode_t uplo,
                                int               n,
                                int               k,
                                const hipComplex* alpha,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* B,
                                int               ldb,
                                const hipComplex* beta,
                                hipComplex*       C,
                                int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasChemm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhemm_v2(hipblasHandle_t         handle,
                                hipblasSideMode_t       side,
                                hipblasFillMode_t       uplo,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* B,
                                int                     ldb,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZhemm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
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
catch(...)
{
    return exception_to_hipblas_status();
}

// hemm_batched
hipblasStatus_t hipblasChemmBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChemmBatched_v2(hipblasHandle_t         handle,
                                       hipblasSideMode_t       side,
                                       hipblasFillMode_t       uplo,
                                       int                     n,
                                       int                     k,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const B[],
                                       int                     ldb,
                                       const hipComplex*       beta,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmBatched_v2(hipblasHandle_t               handle,
                                       hipblasSideMode_t             side,
                                       hipblasFillMode_t             uplo,
                                       int                           n,
                                       int                           k,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const B[],
                                       int                           ldb,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// hemm_strided_batched
hipblasStatus_t hipblasChemmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasChemmStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasSideMode_t side,
                                              hipblasFillMode_t uplo,
                                              int               n,
                                              int               k,
                                              const hipComplex* alpha,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     strideA,
                                              const hipComplex* B,
                                              int               ldb,
                                              hipblasStride     strideB,
                                              const hipComplex* beta,
                                              hipComplex*       C,
                                              int               ldc,
                                              hipblasStride     strideC,
                                              int               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZhemmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasSideMode_t       side,
                                              hipblasFillMode_t       uplo,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* B,
                                              int                     ldb,
                                              hipblasStride           strideB,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

//  trmm
hipblasStatus_t hipblasStrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const float*       alpha,
                             const float*       A,
                             int                lda,
                             const float*       B,
                             int                ldb,
                             float*             C,
                             int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStrmm((cublasHandle_t)handle,
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
                                                  ldb,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmm(hipblasHandle_t    handle,
                             hipblasSideMode_t  side,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                n,
                             const double*      alpha,
                             const double*      A,
                             int                lda,
                             const double*      B,
                             int                ldb,
                             double*            C,
                             int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtrmm((cublasHandle_t)handle,
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
                                                  ldb,
                                                  C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* B,
                             int                   ldb,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* B,
                             int                         ldb,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmm_v2(hipblasHandle_t    handle,
                                hipblasSideMode_t  side,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                m,
                                int                n,
                                const hipComplex*  alpha,
                                const hipComplex*  A,
                                int                lda,
                                const hipComplex*  B,
                                int                ldb,
                                hipComplex*        C,
                                int                ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmm_v2(hipblasHandle_t         handle,
                                hipblasSideMode_t       side,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* B,
                                int                     ldb,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

//  trmmBatched
hipblasStatus_t hipblasStrmmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const float*       alpha,
                                    const float* const A[],
                                    int                lda,
                                    const float* const B[],
                                    int                ldb,
                                    float* const       C[],
                                    int                ldc,
                                    int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const B[],
                                    int                 ldb,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const B[],
                                    int                         ldb,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const B[],
                                    int                               ldb,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmBatched_v2(hipblasHandle_t         handle,
                                       hipblasSideMode_t       side,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const B[],
                                       int                     ldb,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmBatched_v2(hipblasHandle_t               handle,
                                       hipblasSideMode_t             side,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const B[],
                                       int                           ldb,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

//  trmmStridedBatched
hipblasStatus_t hipblasStrmmStridedBatched(hipblasHandle_t    handle,
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
                                           const float*       B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           float*             C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrmmStridedBatched(hipblasHandle_t    handle,
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
                                           const double*      B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           double*            C,
                                           int                ldc,
                                           hipblasStride      strideC,
                                           int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmStridedBatched(hipblasHandle_t       handle,
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
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           hipblasStride         strideC,
                                           int                   batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmStridedBatched(hipblasHandle_t             handle,
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
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           hipblasStride               strideC,
                                           int                         batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrmmStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasSideMode_t  side,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                m,
                                              int                n,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              const hipComplex*  B,
                                              int                ldb,
                                              hipblasStride      strideB,
                                              hipComplex*        C,
                                              int                ldc,
                                              hipblasStride      strideC,
                                              int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrmmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasSideMode_t       side,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              const hipDoubleComplex* B,
                                              int                     ldb,
                                              hipblasStride           strideB,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           strideC,
                                              int                     batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
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
                             const float*       A,
                             int                lda,
                             float*             B,
                             int                ldb)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsm(hipblasHandle_t    handle,
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
                             int                ldb)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsm(hipblasHandle_t       handle,
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
                             int                   ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsm(hipblasHandle_t             handle,
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
                             int                         ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsm_v2(hipblasHandle_t    handle,
                                hipblasSideMode_t  side,
                                hipblasFillMode_t  uplo,
                                hipblasOperation_t transA,
                                hipblasDiagType_t  diag,
                                int                m,
                                int                n,
                                const hipComplex*  alpha,
                                const hipComplex*  A,
                                int                lda,
                                hipComplex*        B,
                                int                ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuComplex*)alpha,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)B,
                                                  ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsm_v2(hipblasHandle_t         handle,
                                hipblasSideMode_t       side,
                                hipblasFillMode_t       uplo,
                                hipblasOperation_t      transA,
                                hipblasDiagType_t       diag,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                hipDoubleComplex*       B,
                                int                     ldb)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  hipFillToCudaFill(uplo),
                                                  hipOperationToCudaOperation(transA),
                                                  hipDiagonalToCudaDiagonal(diag),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)alpha,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)B,
                                                  ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_batched
hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t    handle,
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
                                    int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasStrsmBatched((cublasHandle_t)handle,
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
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t     handle,
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
                                    int                 batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDtrsmBatched((cublasHandle_t)handle,
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
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t             handle,
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
                                    int                         batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsmBatched((cublasHandle_t)handle,
                                                         hipSideToCudaSide(side),
                                                         hipFillToCudaFill(uplo),
                                                         hipOperationToCudaOperation(transA),
                                                         hipDiagonalToCudaDiagonal(diag),
                                                         m,
                                                         n,
                                                         (cuComplex*)alpha,
                                                         (cuComplex**)A,
                                                         lda,
                                                         (cuComplex**)B,
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t                   handle,
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
                                    int                               batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsmBatched((cublasHandle_t)handle,
                                                         hipSideToCudaSide(side),
                                                         hipFillToCudaFill(uplo),
                                                         hipOperationToCudaOperation(transA),
                                                         hipDiagonalToCudaDiagonal(diag),
                                                         m,
                                                         n,
                                                         (cuDoubleComplex*)alpha,
                                                         (cuDoubleComplex**)A,
                                                         lda,
                                                         (cuDoubleComplex**)B,
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmBatched_v2(hipblasHandle_t         handle,
                                       hipblasSideMode_t       side,
                                       hipblasFillMode_t       uplo,
                                       hipblasOperation_t      transA,
                                       hipblasDiagType_t       diag,
                                       int                     m,
                                       int                     n,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       hipComplex* const       B[],
                                       int                     ldb,
                                       int                     batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCtrsmBatched((cublasHandle_t)handle,
                                                         hipSideToCudaSide(side),
                                                         hipFillToCudaFill(uplo),
                                                         hipOperationToCudaOperation(transA),
                                                         hipDiagonalToCudaDiagonal(diag),
                                                         m,
                                                         n,
                                                         (cuComplex*)alpha,
                                                         (cuComplex**)A,
                                                         lda,
                                                         (cuComplex**)B,
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmBatched_v2(hipblasHandle_t               handle,
                                       hipblasSideMode_t             side,
                                       hipblasFillMode_t             uplo,
                                       hipblasOperation_t            transA,
                                       hipblasDiagType_t             diag,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       hipDoubleComplex* const       B[],
                                       int                           ldb,
                                       int                           batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZtrsmBatched((cublasHandle_t)handle,
                                                         hipSideToCudaSide(side),
                                                         hipFillToCudaFill(uplo),
                                                         hipOperationToCudaOperation(transA),
                                                         hipDiagonalToCudaDiagonal(diag),
                                                         m,
                                                         n,
                                                         (cuDoubleComplex*)alpha,
                                                         (cuDoubleComplex**)A,
                                                         lda,
                                                         (cuDoubleComplex**)B,
                                                         ldb,
                                                         batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_strided_batched
hipblasStatus_t hipblasStrsmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrsmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrsmStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasSideMode_t  side,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              hipblasDiagType_t  diag,
                                              int                m,
                                              int                n,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              hipblasStride      strideA,
                                              hipComplex*        B,
                                              int                ldb,
                                              hipblasStride      strideB,
                                              int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrsmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasSideMode_t       side,
                                              hipblasFillMode_t       uplo,
                                              hipblasOperation_t      transA,
                                              hipblasDiagType_t       diag,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           strideA,
                                              hipDoubleComplex*       B,
                                              int                     ldb,
                                              hipblasStride           strideB,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trtri
hipblasStatus_t hipblasStrtri(hipblasHandle_t   handle,
                              hipblasFillMode_t uplo,
                              hipblasDiagType_t diag,
                              int               n,
                              const float*      A,
                              int               lda,
                              float*            invA,
                              int               ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrtri(hipblasHandle_t   handle,
                              hipblasFillMode_t uplo,
                              hipblasDiagType_t diag,
                              int               n,
                              const double*     A,
                              int               lda,
                              double*           invA,
                              int               ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtri(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasDiagType_t     diag,
                              int                   n,
                              const hipblasComplex* A,
                              int                   lda,
                              hipblasComplex*       invA,
                              int                   ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtri(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasDiagType_t           diag,
                              int                         n,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              hipblasDoubleComplex*       invA,
                              int                         ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtri_v2(hipblasHandle_t   handle,
                                 hipblasFillMode_t uplo,
                                 hipblasDiagType_t diag,
                                 int               n,
                                 const hipComplex* A,
                                 int               lda,
                                 hipComplex*       invA,
                                 int               ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtri_v2(hipblasHandle_t         handle,
                                 hipblasFillMode_t       uplo,
                                 hipblasDiagType_t       diag,
                                 int                     n,
                                 const hipDoubleComplex* A,
                                 int                     lda,
                                 hipDoubleComplex*       invA,
                                 int                     ldinvA)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trtri_batched
hipblasStatus_t hipblasStrtriBatched(hipblasHandle_t    handle,
                                     hipblasFillMode_t  uplo,
                                     hipblasDiagType_t  diag,
                                     int                n,
                                     const float* const A[],
                                     int                lda,
                                     float*             invA[],
                                     int                ldinvA,
                                     int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrtriBatched(hipblasHandle_t     handle,
                                     hipblasFillMode_t   uplo,
                                     hipblasDiagType_t   diag,
                                     int                 n,
                                     const double* const A[],
                                     int                 lda,
                                     double*             invA[],
                                     int                 ldinvA,
                                     int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtriBatched(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasDiagType_t           diag,
                                     int                         n,
                                     const hipblasComplex* const A[],
                                     int                         lda,
                                     hipblasComplex*             invA[],
                                     int                         ldinvA,
                                     int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtriBatched(hipblasHandle_t                   handle,
                                     hipblasFillMode_t                 uplo,
                                     hipblasDiagType_t                 diag,
                                     int                               n,
                                     const hipblasDoubleComplex* const A[],
                                     int                               lda,
                                     hipblasDoubleComplex*             invA[],
                                     int                               ldinvA,
                                     int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtriBatched_v2(hipblasHandle_t         handle,
                                        hipblasFillMode_t       uplo,
                                        hipblasDiagType_t       diag,
                                        int                     n,
                                        const hipComplex* const A[],
                                        int                     lda,
                                        hipComplex*             invA[],
                                        int                     ldinvA,
                                        int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtriBatched_v2(hipblasHandle_t               handle,
                                        hipblasFillMode_t             uplo,
                                        hipblasDiagType_t             diag,
                                        int                           n,
                                        const hipDoubleComplex* const A[],
                                        int                           lda,
                                        hipDoubleComplex*             invA[],
                                        int                           ldinvA,
                                        int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// trtri_strided_batched
hipblasStatus_t hipblasStrtriStridedBatched(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            hipblasDiagType_t diag,
                                            int               n,
                                            const float*      A,
                                            int               lda,
                                            hipblasStride     stride_A,
                                            float*            invA,
                                            int               ldinvA,
                                            hipblasStride     stride_invA,
                                            int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDtrtriStridedBatched(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            hipblasDiagType_t diag,
                                            int               n,
                                            const double*     A,
                                            int               lda,
                                            hipblasStride     stride_A,
                                            double*           invA,
                                            int               ldinvA,
                                            hipblasStride     stride_invA,
                                            int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtriStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasDiagType_t     diag,
                                            int                   n,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            hipblasStride         stride_A,
                                            hipblasComplex*       invA,
                                            int                   ldinvA,
                                            hipblasStride         stride_invA,
                                            int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtriStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasDiagType_t           diag,
                                            int                         n,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            hipblasStride               stride_A,
                                            hipblasDoubleComplex*       invA,
                                            int                         ldinvA,
                                            hipblasStride               stride_invA,
                                            int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCtrtriStridedBatched_v2(hipblasHandle_t   handle,
                                               hipblasFillMode_t uplo,
                                               hipblasDiagType_t diag,
                                               int               n,
                                               const hipComplex* A,
                                               int               lda,
                                               hipblasStride     stride_A,
                                               hipComplex*       invA,
                                               int               ldinvA,
                                               hipblasStride     stride_invA,
                                               int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZtrtriStridedBatched_v2(hipblasHandle_t         handle,
                                               hipblasFillMode_t       uplo,
                                               hipblasDiagType_t       diag,
                                               int                     n,
                                               const hipDoubleComplex* A,
                                               int                     lda,
                                               hipblasStride           stride_A,
                                               hipDoubleComplex*       invA,
                                               int                     ldinvA,
                                               hipblasStride           stride_invA,
                                               int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dgmm
hipblasStatus_t hipblasSdgmm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             int               m,
                             int               n,
                             const float*      A,
                             int               lda,
                             const float*      x,
                             int               incx,
                             float*            C,
                             int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSdgmm(
        (cublasHandle_t)handle, hipSideToCudaSide(side), m, n, A, lda, x, incx, C, ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdgmm(hipblasHandle_t   handle,
                             hipblasSideMode_t side,
                             int               m,
                             int               n,
                             const double*     A,
                             int               lda,
                             const double*     x,
                             int               incx,
                             double*           C,
                             int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDdgmm(
        (cublasHandle_t)handle, hipSideToCudaSide(side), m, n, A, lda, x, incx, C, ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdgmm(hipblasHandle_t       handle,
                             hipblasSideMode_t     side,
                             int                   m,
                             int                   n,
                             const hipblasComplex* A,
                             int                   lda,
                             const hipblasComplex* x,
                             int                   incx,
                             hipblasComplex*       C,
                             int                   ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdgmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  m,
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdgmm(hipblasHandle_t             handle,
                             hipblasSideMode_t           side,
                             int                         m,
                             int                         n,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             hipblasDoubleComplex*       C,
                             int                         ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdgmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdgmm_v2(hipblasHandle_t   handle,
                                hipblasSideMode_t side,
                                int               m,
                                int               n,
                                const hipComplex* A,
                                int               lda,
                                const hipComplex* x,
                                int               incx,
                                hipComplex*       C,
                                int               ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCdgmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  m,
                                                  n,
                                                  (cuComplex*)A,
                                                  lda,
                                                  (cuComplex*)x,
                                                  incx,
                                                  (cuComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdgmm_v2(hipblasHandle_t         handle,
                                hipblasSideMode_t       side,
                                int                     m,
                                int                     n,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* x,
                                int                     incx,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZdgmm((cublasHandle_t)handle,
                                                  hipSideToCudaSide(side),
                                                  m,
                                                  n,
                                                  (cuDoubleComplex*)A,
                                                  lda,
                                                  (cuDoubleComplex*)x,
                                                  incx,
                                                  (cuDoubleComplex*)C,
                                                  ldc));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// dgmm_batched
hipblasStatus_t hipblasSdgmmBatched(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    int                m,
                                    int                n,
                                    const float* const A[],
                                    int                lda,
                                    const float* const x[],
                                    int                incx,
                                    float* const       C[],
                                    int                ldc,
                                    int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdgmmBatched(hipblasHandle_t     handle,
                                    hipblasSideMode_t   side,
                                    int                 m,
                                    int                 n,
                                    const double* const A[],
                                    int                 lda,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       C[],
                                    int                 ldc,
                                    int                 batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdgmmBatched(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    int                         m,
                                    int                         n,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       C[],
                                    int                         ldc,
                                    int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdgmmBatched(hipblasHandle_t                   handle,
                                    hipblasSideMode_t                 side,
                                    int                               m,
                                    int                               n,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       C[],
                                    int                               ldc,
                                    int                               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdgmmBatched_v2(hipblasHandle_t         handle,
                                       hipblasSideMode_t       side,
                                       int                     m,
                                       int                     n,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const x[],
                                       int                     incx,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdgmmBatched_v2(hipblasHandle_t               handle,
                                       hipblasSideMode_t             side,
                                       int                           m,
                                       int                           n,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const x[],
                                       int                           incx,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dgmm_strided_batched
hipblasStatus_t hipblasSdgmmStridedBatched(hipblasHandle_t   handle,
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
                                           int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDdgmmStridedBatched(hipblasHandle_t   handle,
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
                                           int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdgmmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdgmmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCdgmmStridedBatched_v2(hipblasHandle_t   handle,
                                              hipblasSideMode_t side,
                                              int               m,
                                              int               n,
                                              const hipComplex* A,
                                              int               lda,
                                              hipblasStride     stride_A,
                                              const hipComplex* x,
                                              int               incx,
                                              hipblasStride     stride_x,
                                              hipComplex*       C,
                                              int               ldc,
                                              hipblasStride     stride_C,
                                              int               batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZdgmmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasSideMode_t       side,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              hipblasStride           stride_A,
                                              const hipDoubleComplex* x,
                                              int                     incx,
                                              hipblasStride           stride_x,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              hipblasStride           stride_C,
                                              int                     batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

#ifdef __HIP_PLATFORM_SOLVER__

// getrf
hipblasStatus_t hipblasSgetrf(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetrf(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrf(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrf(hipblasHandle_t       handle,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              int*                  ipiv,
                              int*                  info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrf_v2(
    hipblasHandle_t handle, const int n, hipComplex* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrf_v2(
    hipblasHandle_t handle, const int n, hipDoubleComplex* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// getrf_batched
hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     int*            ipiv,
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSgetrfBatched((cublasHandle_t)handle, n, A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     int*            ipiv,
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDgetrfBatched((cublasHandle_t)handle, n, A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetrfBatched(hipblasHandle_t       handle,
                                     const int             n,
                                     hipblasComplex* const A[],
                                     const int             lda,
                                     int*                  ipiv,
                                     int*                  info,
                                     const int             batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetrfBatched(
        (cublasHandle_t)handle, n, (cuComplex**)A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t             handle,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     int*                        ipiv,
                                     int*                        info,
                                     const int                   batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetrfBatched(
        (cublasHandle_t)handle, n, (cuDoubleComplex**)A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetrfBatched_v2(hipblasHandle_t   handle,
                                        const int         n,
                                        hipComplex* const A[],
                                        const int         lda,
                                        int*              ipiv,
                                        int*              info,
                                        const int         batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetrfBatched(
        (cublasHandle_t)handle, n, (cuComplex**)A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetrfBatched_v2(hipblasHandle_t         handle,
                                        const int               n,
                                        hipDoubleComplex* const A[],
                                        const int               lda,
                                        int*                    ipiv,
                                        int*                    info,
                                        const int               batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetrfBatched(
        (cublasHandle_t)handle, n, (cuDoubleComplex**)A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// getrf_strided_batched
hipblasStatus_t hipblasSgetrfStridedBatched(hipblasHandle_t     handle,
                                            const int           n,
                                            float*              A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            int*                ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetrfStridedBatched(hipblasHandle_t     handle,
                                            const int           n,
                                            double*             A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            int*                ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrfStridedBatched(hipblasHandle_t     handle,
                                            const int           n,
                                            hipblasComplex*     A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            int*                ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrfStridedBatched(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasDoubleComplex* A,
                                            const int             lda,
                                            const hipblasStride   strideA,
                                            int*                  ipiv,
                                            const hipblasStride   strideP,
                                            int*                  info,
                                            const int             batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrfStridedBatched_v2(hipblasHandle_t     handle,
                                               const int           n,
                                               hipComplex*         A,
                                               const int           lda,
                                               const hipblasStride strideA,
                                               int*                ipiv,
                                               const hipblasStride strideP,
                                               int*                info,
                                               const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrfStridedBatched_v2(hipblasHandle_t     handle,
                                               const int           n,
                                               hipDoubleComplex*   A,
                                               const int           lda,
                                               const hipblasStride strideA,
                                               int*                ipiv,
                                               const hipblasStride strideP,
                                               int*                info,
                                               const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// getrs
hipblasStatus_t hipblasSgetrs(hipblasHandle_t          handle,
                              const hipblasOperation_t trans,
                              const int                n,
                              const int                nrhs,
                              float*                   A,
                              const int                lda,
                              const int*               ipiv,
                              float*                   B,
                              const int                ldb,
                              int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetrs(hipblasHandle_t          handle,
                              const hipblasOperation_t trans,
                              const int                n,
                              const int                nrhs,
                              double*                  A,
                              const int                lda,
                              const int*               ipiv,
                              double*                  B,
                              const int                ldb,
                              int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrs(hipblasHandle_t          handle,
                              const hipblasOperation_t trans,
                              const int                n,
                              const int                nrhs,
                              hipblasComplex*          A,
                              const int                lda,
                              const int*               ipiv,
                              hipblasComplex*          B,
                              const int                ldb,
                              int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrs(hipblasHandle_t          handle,
                              const hipblasOperation_t trans,
                              const int                n,
                              const int                nrhs,
                              hipblasDoubleComplex*    A,
                              const int                lda,
                              const int*               ipiv,
                              hipblasDoubleComplex*    B,
                              const int                ldb,
                              int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrs_v2(hipblasHandle_t          handle,
                                 const hipblasOperation_t trans,
                                 const int                n,
                                 const int                nrhs,
                                 hipComplex*              A,
                                 const int                lda,
                                 const int*               ipiv,
                                 hipComplex*              B,
                                 const int                ldb,
                                 int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrs_v2(hipblasHandle_t          handle,
                                 const hipblasOperation_t trans,
                                 const int                n,
                                 const int                nrhs,
                                 hipDoubleComplex*        A,
                                 const int                lda,
                                 const int*               ipiv,
                                 hipDoubleComplex*        B,
                                 const int                ldb,
                                 int*                     info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// getrs_batched
hipblasStatus_t hipblasSgetrsBatched(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     float* const             A[],
                                     const int                lda,
                                     const int*               ipiv,
                                     float* const             B[],
                                     const int                ldb,
                                     int*                     info,
                                     const int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          A,
                                                          lda,
                                                          ipiv,
                                                          B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double* const            A[],
                                     const int                lda,
                                     const int*               ipiv,
                                     double* const            B[],
                                     const int                ldb,
                                     int*                     info,
                                     const int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          A,
                                                          lda,
                                                          ipiv,
                                                          B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetrsBatched(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     hipblasComplex* const    A[],
                                     const int                lda,
                                     const int*               ipiv,
                                     hipblasComplex* const    B[],
                                     const int                ldb,
                                     int*                     info,
                                     const int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (cuComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuComplex**)B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetrsBatched(hipblasHandle_t             handle,
                                     const hipblasOperation_t    trans,
                                     const int                   n,
                                     const int                   nrhs,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     const int*                  ipiv,
                                     hipblasDoubleComplex* const B[],
                                     const int                   ldb,
                                     int*                        info,
                                     const int                   batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuDoubleComplex**)B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetrsBatched_v2(hipblasHandle_t          handle,
                                        const hipblasOperation_t trans,
                                        const int                n,
                                        const int                nrhs,
                                        hipComplex* const        A[],
                                        const int                lda,
                                        const int*               ipiv,
                                        hipComplex* const        B[],
                                        const int                ldb,
                                        int*                     info,
                                        const int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (cuComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuComplex**)B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetrsBatched_v2(hipblasHandle_t          handle,
                                        const hipblasOperation_t trans,
                                        const int                n,
                                        const int                nrhs,
                                        hipDoubleComplex* const  A[],
                                        const int                lda,
                                        const int*               ipiv,
                                        hipDoubleComplex* const  B[],
                                        const int                ldb,
                                        int*                     info,
                                        const int                batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetrsBatched((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuDoubleComplex**)B,
                                                          ldb,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// getrs_strided_batched
hipblasStatus_t hipblasSgetrsStridedBatched(hipblasHandle_t          handle,
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
                                            const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetrsStridedBatched(hipblasHandle_t          handle,
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
                                            const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrsStridedBatched(hipblasHandle_t          handle,
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
                                            const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrsStridedBatched(hipblasHandle_t          handle,
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
                                            const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgetrsStridedBatched_v2(hipblasHandle_t          handle,
                                               const hipblasOperation_t trans,
                                               const int                n,
                                               const int                nrhs,
                                               hipComplex*              A,
                                               const int                lda,
                                               const hipblasStride      strideA,
                                               const int*               ipiv,
                                               const hipblasStride      strideP,
                                               hipComplex*              B,
                                               const int                ldb,
                                               const hipblasStride      strideB,
                                               int*                     info,
                                               const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgetrsStridedBatched_v2(hipblasHandle_t          handle,
                                               const hipblasOperation_t trans,
                                               const int                n,
                                               const int                nrhs,
                                               hipDoubleComplex*        A,
                                               const int                lda,
                                               const hipblasStride      strideA,
                                               const int*               ipiv,
                                               const hipblasStride      strideP,
                                               hipDoubleComplex*        B,
                                               const int                ldb,
                                               const hipblasStride      strideB,
                                               int*                     info,
                                               const int                batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// getri_batched
hipblasStatus_t hipblasSgetriBatched(hipblasHandle_t handle,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     int*            ipiv,
                                     float* const    C[],
                                     const int       ldc,
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSgetriBatched((cublasHandle_t)handle, n, A, lda, ipiv, C, ldc, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     int*            ipiv,
                                     double* const   C[],
                                     const int       ldc,
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDgetriBatched((cublasHandle_t)handle, n, A, lda, ipiv, C, ldc, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetriBatched(hipblasHandle_t       handle,
                                     const int             n,
                                     hipblasComplex* const A[],
                                     const int             lda,
                                     int*                  ipiv,
                                     hipblasComplex* const C[],
                                     const int             ldc,
                                     int*                  info,
                                     const int             batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetriBatched((cublasHandle_t)handle,
                                                          n,
                                                          (cuComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuComplex**)C,
                                                          ldc,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t             handle,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     int*                        ipiv,
                                     hipblasDoubleComplex* const C[],
                                     const int                   ldc,
                                     int*                        info,
                                     const int                   batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetriBatched((cublasHandle_t)handle,
                                                          n,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuDoubleComplex**)C,
                                                          ldc,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetriBatched_v2(hipblasHandle_t   handle,
                                        const int         n,
                                        hipComplex* const A[],
                                        const int         lda,
                                        int*              ipiv,
                                        hipComplex* const C[],
                                        const int         ldc,
                                        int*              info,
                                        const int         batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgetriBatched((cublasHandle_t)handle,
                                                          n,
                                                          (cuComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuComplex**)C,
                                                          ldc,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetriBatched_v2(hipblasHandle_t         handle,
                                        const int               n,
                                        hipDoubleComplex* const A[],
                                        const int               lda,
                                        int*                    ipiv,
                                        hipDoubleComplex* const C[],
                                        const int               ldc,
                                        int*                    info,
                                        const int               batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgetriBatched((cublasHandle_t)handle,
                                                          n,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          ipiv,
                                                          (cuDoubleComplex**)C,
                                                          ldc,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// geqrf
hipblasStatus_t hipblasSgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              float*          A,
                              const int       lda,
                              float*          ipiv,
                              int*            info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              double*         A,
                              const int       lda,
                              double*         ipiv,
                              int*            info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              hipblasComplex* A,
                              const int       lda,
                              hipblasComplex* ipiv,
                              int*            info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeqrf(hipblasHandle_t       handle,
                              const int             m,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              hipblasDoubleComplex* ipiv,
                              int*                  info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeqrf_v2(hipblasHandle_t handle,
                                 const int       m,
                                 const int       n,
                                 hipComplex*     A,
                                 const int       lda,
                                 hipComplex*     ipiv,
                                 int*            info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeqrf_v2(hipblasHandle_t   handle,
                                 const int         m,
                                 const int         n,
                                 hipDoubleComplex* A,
                                 const int         lda,
                                 hipDoubleComplex* ipiv,
                                 int*              info)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     float* const    ipiv[],
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasSgeqrfBatched((cublasHandle_t)handle, m, n, A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     double* const   ipiv[],
                                     int*            info,
                                     const int       batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasDgeqrfBatched((cublasHandle_t)handle, m, n, A, lda, ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeqrfBatched(hipblasHandle_t       handle,
                                     const int             m,
                                     const int             n,
                                     hipblasComplex* const A[],
                                     const int             lda,
                                     hipblasComplex* const ipiv[],
                                     int*                  info,
                                     const int             batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeqrfBatched(
        (cublasHandle_t)handle, m, n, (cuComplex**)A, lda, (cuComplex**)ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeqrfBatched(hipblasHandle_t             handle,
                                     const int                   m,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     hipblasDoubleComplex* const ipiv[],
                                     int*                        info,
                                     const int                   batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeqrfBatched((cublasHandle_t)handle,
                                                          m,
                                                          n,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          (cuDoubleComplex**)ipiv,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeqrfBatched_v2(hipblasHandle_t   handle,
                                        const int         m,
                                        const int         n,
                                        hipComplex* const A[],
                                        const int         lda,
                                        hipComplex* const ipiv[],
                                        int*              info,
                                        const int         batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgeqrfBatched(
        (cublasHandle_t)handle, m, n, (cuComplex**)A, lda, (cuComplex**)ipiv, info, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeqrfBatched_v2(hipblasHandle_t         handle,
                                        const int               m,
                                        const int               n,
                                        hipDoubleComplex* const A[],
                                        const int               lda,
                                        hipDoubleComplex* const ipiv[],
                                        int*                    info,
                                        const int               batch_count)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgeqrfBatched((cublasHandle_t)handle,
                                                          m,
                                                          n,
                                                          (cuDoubleComplex**)A,
                                                          lda,
                                                          (cuDoubleComplex**)ipiv,
                                                          info,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// geqrf_strided_batched
hipblasStatus_t hipblasSgeqrfStridedBatched(hipblasHandle_t     handle,
                                            const int           m,
                                            const int           n,
                                            float*              A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            float*              ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgeqrfStridedBatched(hipblasHandle_t     handle,
                                            const int           m,
                                            const int           n,
                                            double*             A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            double*             ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeqrfStridedBatched(hipblasHandle_t     handle,
                                            const int           m,
                                            const int           n,
                                            hipblasComplex*     A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            hipblasComplex*     ipiv,
                                            const hipblasStride strideP,
                                            int*                info,
                                            const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeqrfStridedBatched(hipblasHandle_t       handle,
                                            const int             m,
                                            const int             n,
                                            hipblasDoubleComplex* A,
                                            const int             lda,
                                            const hipblasStride   strideA,
                                            hipblasDoubleComplex* ipiv,
                                            const hipblasStride   strideP,
                                            int*                  info,
                                            const int             batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgeqrfStridedBatched_v2(hipblasHandle_t     handle,
                                               const int           m,
                                               const int           n,
                                               hipComplex*         A,
                                               const int           lda,
                                               const hipblasStride strideA,
                                               hipComplex*         ipiv,
                                               const hipblasStride strideP,
                                               int*                info,
                                               const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgeqrfStridedBatched_v2(hipblasHandle_t     handle,
                                               const int           m,
                                               const int           n,
                                               hipDoubleComplex*   A,
                                               const int           lda,
                                               const hipblasStride strideA,
                                               hipDoubleComplex*   ipiv,
                                               const hipblasStride strideP,
                                               int*                info,
                                               const int           batch_count)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gels
hipblasStatus_t hipblasSgels(hipblasHandle_t    handle,
                             hipblasOperation_t trans,
                             const int          m,
                             const int          n,
                             const int          nrhs,
                             float*             A,
                             const int          lda,
                             float*             B,
                             const int          ldb,
                             int*               info,
                             int*               deviceInfo)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgels(hipblasHandle_t    handle,
                             hipblasOperation_t trans,
                             const int          m,
                             const int          n,
                             const int          nrhs,
                             double*            A,
                             const int          lda,
                             double*            B,
                             const int          ldb,
                             int*               info,
                             int*               deviceInfo)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgels(hipblasHandle_t    handle,
                             hipblasOperation_t trans,
                             const int          m,
                             const int          n,
                             const int          nrhs,
                             hipblasComplex*    A,
                             const int          lda,
                             hipblasComplex*    B,
                             const int          ldb,
                             int*               info,
                             int*               deviceInfo)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgels(hipblasHandle_t       handle,
                             hipblasOperation_t    trans,
                             const int             m,
                             const int             n,
                             const int             nrhs,
                             hipblasDoubleComplex* A,
                             const int             lda,
                             hipblasDoubleComplex* B,
                             const int             ldb,
                             int*                  info,
                             int*                  deviceInfo)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgels_v2(hipblasHandle_t    handle,
                                hipblasOperation_t trans,
                                const int          m,
                                const int          n,
                                const int          nrhs,
                                hipComplex*        A,
                                const int          lda,
                                hipComplex*        B,
                                const int          ldb,
                                int*               info,
                                int*               deviceInfo)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgels_v2(hipblasHandle_t    handle,
                                hipblasOperation_t trans,
                                const int          m,
                                const int          n,
                                const int          nrhs,
                                hipDoubleComplex*  A,
                                const int          lda,
                                hipDoubleComplex*  B,
                                const int          ldb,
                                int*               info,
                                int*               deviceInfo)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// gelsBatched
hipblasStatus_t hipblasSgelsBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    float* const       A[],
                                    const int          lda,
                                    float* const       B[],
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo,
                                    const int          batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasSgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         A,
                                                         lda,
                                                         B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgelsBatched(hipblasHandle_t    handle,
                                    hipblasOperation_t trans,
                                    const int          m,
                                    const int          n,
                                    const int          nrhs,
                                    double* const      A[],
                                    const int          lda,
                                    double* const      B[],
                                    const int          ldb,
                                    int*               info,
                                    int*               deviceInfo,
                                    const int          batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         A,
                                                         lda,
                                                         B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgelsBatched(hipblasHandle_t       handle,
                                    hipblasOperation_t    trans,
                                    const int             m,
                                    const int             n,
                                    const int             nrhs,
                                    hipblasComplex* const A[],
                                    const int             lda,
                                    hipblasComplex* const B[],
                                    const int             ldb,
                                    int*                  info,
                                    int*                  deviceInfo,
                                    const int             batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         (cuComplex**)A,
                                                         lda,
                                                         (cuComplex**)B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgelsBatched(hipblasHandle_t             handle,
                                    hipblasOperation_t          trans,
                                    const int                   m,
                                    const int                   n,
                                    const int                   nrhs,
                                    hipblasDoubleComplex* const A[],
                                    const int                   lda,
                                    hipblasDoubleComplex* const B[],
                                    const int                   ldb,
                                    int*                        info,
                                    int*                        deviceInfo,
                                    const int                   batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         (cuDoubleComplex**)A,
                                                         lda,
                                                         (cuDoubleComplex**)B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgelsBatched_v2(hipblasHandle_t    handle,
                                       hipblasOperation_t trans,
                                       const int          m,
                                       const int          n,
                                       const int          nrhs,
                                       hipComplex* const  A[],
                                       const int          lda,
                                       hipComplex* const  B[],
                                       const int          ldb,
                                       int*               info,
                                       int*               deviceInfo,
                                       const int          batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         (cuComplex**)A,
                                                         lda,
                                                         (cuComplex**)B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgelsBatched_v2(hipblasHandle_t         handle,
                                       hipblasOperation_t      trans,
                                       const int               m,
                                       const int               n,
                                       const int               nrhs,
                                       hipDoubleComplex* const A[],
                                       const int               lda,
                                       hipDoubleComplex* const B[],
                                       const int               ldb,
                                       int*                    info,
                                       int*                    deviceInfo,
                                       const int               batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgelsBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(trans),
                                                         m,
                                                         n,
                                                         nrhs,
                                                         (cuDoubleComplex**)A,
                                                         lda,
                                                         (cuDoubleComplex**)B,
                                                         ldb,
                                                         info,
                                                         deviceInfo,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gelsStridedBatched
hipblasStatus_t hipblasSgelsStridedBatched(hipblasHandle_t     handle,
                                           hipblasOperation_t  trans,
                                           const int           m,
                                           const int           n,
                                           const int           nrhs,
                                           float*              A,
                                           const int           lda,
                                           const hipblasStride strideA,
                                           float*              B,
                                           const int           ldb,
                                           const hipblasStride strideB,
                                           int*                info,
                                           int*                deviceInfo,
                                           const int           batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgelsStridedBatched(hipblasHandle_t     handle,
                                           hipblasOperation_t  trans,
                                           const int           m,
                                           const int           n,
                                           const int           nrhs,
                                           double*             A,
                                           const int           lda,
                                           const hipblasStride strideA,
                                           double*             B,
                                           const int           ldb,
                                           const hipblasStride strideB,
                                           int*                info,
                                           int*                deviceInfo,
                                           const int           batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgelsStridedBatched(hipblasHandle_t     handle,
                                           hipblasOperation_t  trans,
                                           const int           m,
                                           const int           n,
                                           const int           nrhs,
                                           hipblasComplex*     A,
                                           const int           lda,
                                           const hipblasStride strideA,
                                           hipblasComplex*     B,
                                           const int           ldb,
                                           const hipblasStride strideB,
                                           int*                info,
                                           int*                deviceInfo,
                                           const int           batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgelsStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           const int             m,
                                           const int             n,
                                           const int             nrhs,
                                           hipblasDoubleComplex* A,
                                           const int             lda,
                                           const hipblasStride   strideA,
                                           hipblasDoubleComplex* B,
                                           const int             ldb,
                                           const hipblasStride   strideB,
                                           int*                  info,
                                           int*                  deviceInfo,
                                           const int             batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasCgelsStridedBatched_v2(hipblasHandle_t     handle,
                                              hipblasOperation_t  trans,
                                              const int           m,
                                              const int           n,
                                              const int           nrhs,
                                              hipComplex*         A,
                                              const int           lda,
                                              const hipblasStride strideA,
                                              hipComplex*         B,
                                              const int           ldb,
                                              const hipblasStride strideB,
                                              int*                info,
                                              int*                deviceInfo,
                                              const int           batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasZgelsStridedBatched_v2(hipblasHandle_t     handle,
                                              hipblasOperation_t  trans,
                                              const int           m,
                                              const int           n,
                                              const int           nrhs,
                                              hipDoubleComplex*   A,
                                              const int           lda,
                                              const hipblasStride strideA,
                                              hipDoubleComplex*   B,
                                              const int           ldb,
                                              const hipblasStride strideB,
                                              int*                info,
                                              int*                deviceInfo,
                                              const int           batchCount)
{
    // only batched variants of gels are supported in cuBLAS
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

#endif

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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t       handle,
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
                             int                   ldc)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t             handle,
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
                             int                         ldc)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemm_v2(hipblasHandle_t    handle,
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemm_v2(hipblasHandle_t         handle,
                                hipblasOperation_t      transa,
                                hipblasOperation_t      transb,
                                int                     m,
                                int                     n,
                                int                     k,
                                const hipDoubleComplex* alpha,
                                const hipDoubleComplex* A,
                                int                     lda,
                                const hipDoubleComplex* B,
                                int                     ldb,
                                const hipDoubleComplex* beta,
                                hipDoubleComplex*       C,
                                int                     ldc)
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_batched
hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t          handle,
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
                                    int                      batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasHgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         (__half*)alpha,
                                                         (__half* const*)A,
                                                         lda,
                                                         (__half* const*)B,
                                                         ldb,
                                                         (__half*)beta,
                                                         (__half* const*)C,
                                                         ldc,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t             handle,
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
                                    int                         batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         (cuComplex*)alpha,
                                                         (cuComplex* const*)A,
                                                         lda,
                                                         (cuComplex* const*)B,
                                                         ldb,
                                                         (cuComplex*)beta,
                                                         (cuComplex* const*)C,
                                                         ldc,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t                   handle,
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
                                    int                               batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         (cuDoubleComplex*)alpha,
                                                         (cuDoubleComplex* const*)A,
                                                         lda,
                                                         (cuDoubleComplex* const*)B,
                                                         ldb,
                                                         (cuDoubleComplex*)beta,
                                                         (cuDoubleComplex* const*)C,
                                                         ldc,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmBatched_v2(hipblasHandle_t         handle,
                                       hipblasOperation_t      transa,
                                       hipblasOperation_t      transb,
                                       int                     m,
                                       int                     n,
                                       int                     k,
                                       const hipComplex*       alpha,
                                       const hipComplex* const A[],
                                       int                     lda,
                                       const hipComplex* const B[],
                                       int                     ldb,
                                       const hipComplex*       beta,
                                       hipComplex* const       C[],
                                       int                     ldc,
                                       int                     batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         (cuComplex*)alpha,
                                                         (cuComplex* const*)A,
                                                         lda,
                                                         (cuComplex* const*)B,
                                                         ldb,
                                                         (cuComplex*)beta,
                                                         (cuComplex* const*)C,
                                                         ldc,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmBatched_v2(hipblasHandle_t               handle,
                                       hipblasOperation_t            transa,
                                       hipblasOperation_t            transb,
                                       int                           m,
                                       int                           n,
                                       int                           k,
                                       const hipDoubleComplex*       alpha,
                                       const hipDoubleComplex* const A[],
                                       int                           lda,
                                       const hipDoubleComplex* const B[],
                                       int                           ldb,
                                       const hipDoubleComplex*       beta,
                                       hipDoubleComplex* const       C[],
                                       int                           ldc,
                                       int                           batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgemmBatched((cublasHandle_t)handle,
                                                         hipOperationToCudaOperation(transa),
                                                         hipOperationToCudaOperation(transb),
                                                         m,
                                                         n,
                                                         k,
                                                         (cuDoubleComplex*)alpha,
                                                         (cuDoubleComplex* const*)A,
                                                         lda,
                                                         (cuDoubleComplex* const*)B,
                                                         ldb,
                                                         (cuDoubleComplex*)beta,
                                                         (cuDoubleComplex* const*)C,
                                                         ldc,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_strided_batched
hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasHgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                (__half*)alpha,
                                                                (__half*)(A),
                                                                lda,
                                                                bsa,
                                                                (__half*)(B),
                                                                ldb,
                                                                bsb,
                                                                (__half*)beta,
                                                                (__half*)C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

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
try
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
catch(...)
{
    return exception_to_hipblas_status();
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
try
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
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                (cuComplex*)alpha,
                                                                (cuComplex*)(A),
                                                                lda,
                                                                bsa,
                                                                (cuComplex*)(B),
                                                                ldb,
                                                                bsb,
                                                                (cuComplex*)beta,
                                                                (cuComplex*)C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                (cuDoubleComplex*)alpha,
                                                                (cuDoubleComplex*)(A),
                                                                lda,
                                                                bsa,
                                                                (cuDoubleComplex*)(B),
                                                                ldb,
                                                                bsb,
                                                                (cuDoubleComplex*)beta,
                                                                (cuDoubleComplex*)C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgemmStridedBatched_v2(hipblasHandle_t    handle,
                                              hipblasOperation_t transa,
                                              hipblasOperation_t transb,
                                              int                m,
                                              int                n,
                                              int                k,
                                              const hipComplex*  alpha,
                                              const hipComplex*  A,
                                              int                lda,
                                              long long          bsa,
                                              const hipComplex*  B,
                                              int                ldb,
                                              long long          bsb,
                                              const hipComplex*  beta,
                                              hipComplex*        C,
                                              int                ldc,
                                              long long          bsc,
                                              int                batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasCgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                (cuComplex*)alpha,
                                                                (cuComplex*)(A),
                                                                lda,
                                                                bsa,
                                                                (cuComplex*)(B),
                                                                ldb,
                                                                bsb,
                                                                (cuComplex*)beta,
                                                                (cuComplex*)C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgemmStridedBatched_v2(hipblasHandle_t         handle,
                                              hipblasOperation_t      transa,
                                              hipblasOperation_t      transb,
                                              int                     m,
                                              int                     n,
                                              int                     k,
                                              const hipDoubleComplex* alpha,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              long long               bsa,
                                              const hipDoubleComplex* B,
                                              int                     ldb,
                                              long long               bsb,
                                              const hipDoubleComplex* beta,
                                              hipDoubleComplex*       C,
                                              int                     ldc,
                                              long long               bsc,
                                              int                     batchCount)
try
{
    return hipCUBLASStatusToHIPStatus(cublasZgemmStridedBatched((cublasHandle_t)handle,
                                                                hipOperationToCudaOperation(transa),
                                                                hipOperationToCudaOperation(transb),
                                                                m,
                                                                n,
                                                                k,
                                                                (cuDoubleComplex*)alpha,
                                                                (cuDoubleComplex*)(A),
                                                                lda,
                                                                bsa,
                                                                (cuDoubleComplex*)(B),
                                                                ldb,
                                                                bsb,
                                                                (cuDoubleComplex*)beta,
                                                                (cuDoubleComplex*)C,
                                                                ldc,
                                                                bsc,
                                                                batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_ex
hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
                              hipblasOperation_t transa,
                              hipblasOperation_t transb,
                              int                m,
                              int                n,
                              int                k,
                              const void*        alpha,
                              const void*        A,
                              hipblasDatatype_t  a_type,
                              int                lda,
                              const void*        B,
                              hipblasDatatype_t  b_type,
                              int                ldb,
                              const void*        beta,
                              void*              C,
                              hipblasDatatype_t  c_type,
                              int                ldc,
                              hipblasDatatype_t  compute_type,
                              hipblasGemmAlgo_t  algo)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGemmEx((cublasHandle_t)handle,
                                                   hipOperationToCudaOperation(transa),
                                                   hipOperationToCudaOperation(transb),
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   HIPDatatypeToCudaDatatype(a_type),
                                                   lda,
                                                   B,
                                                   HIPDatatypeToCudaDatatype(b_type),
                                                   ldb,
                                                   beta,
                                                   C,
                                                   HIPDatatypeToCudaDatatype(c_type),
                                                   ldc,
                                                   HIPDatatypeToCudaDatatype(compute_type),
                                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmExWithFlags(hipblasHandle_t    handle,
                                       hipblasOperation_t transa,
                                       hipblasOperation_t transb,
                                       int                m,
                                       int                n,
                                       int                k,
                                       const void*        alpha,
                                       const void*        A,
                                       hipblasDatatype_t  a_type,
                                       int                lda,
                                       const void*        B,
                                       hipblasDatatype_t  b_type,
                                       int                ldb,
                                       const void*        beta,
                                       void*              C,
                                       hipblasDatatype_t  c_type,
                                       int                ldc,
                                       hipblasDatatype_t  compute_type,
                                       hipblasGemmAlgo_t  algo,
                                       hipblasGemmFlags_t flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(cublasGemmEx((cublasHandle_t)handle,
                                                   hipOperationToCudaOperation(transa),
                                                   hipOperationToCudaOperation(transb),
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   HIPDatatypeToCudaDatatype(a_type),
                                                   lda,
                                                   B,
                                                   HIPDatatypeToCudaDatatype(b_type),
                                                   ldb,
                                                   beta,
                                                   C,
                                                   HIPDatatypeToCudaDatatype(c_type),
                                                   ldc,
                                                   HIPDatatypeToCudaDatatype(compute_type),
                                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmEx_v2(hipblasHandle_t      handle,
                                 hipblasOperation_t   transa,
                                 hipblasOperation_t   transb,
                                 int                  m,
                                 int                  n,
                                 int                  k,
                                 const void*          alpha,
                                 const void*          A,
                                 hipDataType          a_type,
                                 int                  lda,
                                 const void*          B,
                                 hipDataType          b_type,
                                 int                  ldb,
                                 const void*          beta,
                                 void*                C,
                                 hipDataType          c_type,
                                 int                  ldc,
                                 hipblasComputeType_t compute_type,
                                 hipblasGemmAlgo_t    algo)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGemmEx((cublasHandle_t)handle,
                                                   hipOperationToCudaOperation(transa),
                                                   hipOperationToCudaOperation(transb),
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   HIPDatatypeToCudaDatatype_v2(a_type),
                                                   lda,
                                                   B,
                                                   HIPDatatypeToCudaDatatype_v2(b_type),
                                                   ldb,
                                                   beta,
                                                   C,
                                                   HIPDatatypeToCudaDatatype_v2(c_type),
                                                   ldc,
                                                   HIPComputetypeToCudaComputetype(compute_type),
                                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmExWithFlags_v2(hipblasHandle_t      handle,
                                          hipblasOperation_t   transa,
                                          hipblasOperation_t   transb,
                                          int                  m,
                                          int                  n,
                                          int                  k,
                                          const void*          alpha,
                                          const void*          A,
                                          hipDataType          a_type,
                                          int                  lda,
                                          const void*          B,
                                          hipDataType          b_type,
                                          int                  ldb,
                                          const void*          beta,
                                          void*                C,
                                          hipDataType          c_type,
                                          int                  ldc,
                                          hipblasComputeType_t compute_type,
                                          hipblasGemmAlgo_t    algo,
                                          hipblasGemmFlags_t   flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(cublasGemmEx((cublasHandle_t)handle,
                                                   hipOperationToCudaOperation(transa),
                                                   hipOperationToCudaOperation(transb),
                                                   m,
                                                   n,
                                                   k,
                                                   alpha,
                                                   A,
                                                   HIPDatatypeToCudaDatatype_v2(a_type),
                                                   lda,
                                                   B,
                                                   HIPDatatypeToCudaDatatype_v2(b_type),
                                                   ldb,
                                                   beta,
                                                   C,
                                                   HIPDatatypeToCudaDatatype_v2(c_type),
                                                   ldc,
                                                   HIPComputetypeToCudaComputetype(compute_type),
                                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t    handle,
                                     hipblasOperation_t transa,
                                     hipblasOperation_t transb,
                                     int                m,
                                     int                n,
                                     int                k,
                                     const void*        alpha,
                                     const void*        A[],
                                     hipblasDatatype_t  a_type,
                                     int                lda,
                                     const void*        B[],
                                     hipblasDatatype_t  b_type,
                                     int                ldb,
                                     const void*        beta,
                                     void*              C[],
                                     hipblasDatatype_t  c_type,
                                     int                ldc,
                                     int                batch_count,
                                     hipblasDatatype_t  compute_type,
                                     hipblasGemmAlgo_t  algo)
try
{
    return hipCUBLASStatusToHIPStatus(cublasGemmBatchedEx((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(transa),
                                                          hipOperationToCudaOperation(transb),
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          HIPDatatypeToCudaDatatype(a_type),
                                                          lda,
                                                          B,
                                                          HIPDatatypeToCudaDatatype(b_type),
                                                          ldb,
                                                          beta,
                                                          C,
                                                          HIPDatatypeToCudaDatatype(c_type),
                                                          ldc,
                                                          batch_count,
                                                          HIPDatatypeToCudaDatatype(compute_type),
                                                          HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmBatchedExWithFlags(hipblasHandle_t    handle,
                                              hipblasOperation_t transa,
                                              hipblasOperation_t transb,
                                              int                m,
                                              int                n,
                                              int                k,
                                              const void*        alpha,
                                              const void*        A[],
                                              hipblasDatatype_t  a_type,
                                              int                lda,
                                              const void*        B[],
                                              hipblasDatatype_t  b_type,
                                              int                ldb,
                                              const void*        beta,
                                              void*              C[],
                                              hipblasDatatype_t  c_type,
                                              int                ldc,
                                              int                batch_count,
                                              hipblasDatatype_t  compute_type,
                                              hipblasGemmAlgo_t  algo,
                                              hipblasGemmFlags_t flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(cublasGemmBatchedEx((cublasHandle_t)handle,
                                                          hipOperationToCudaOperation(transa),
                                                          hipOperationToCudaOperation(transb),
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          HIPDatatypeToCudaDatatype(a_type),
                                                          lda,
                                                          B,
                                                          HIPDatatypeToCudaDatatype(b_type),
                                                          ldb,
                                                          beta,
                                                          C,
                                                          HIPDatatypeToCudaDatatype(c_type),
                                                          ldc,
                                                          batch_count,
                                                          HIPDatatypeToCudaDatatype(compute_type),
                                                          HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmBatchedEx_v2(hipblasHandle_t      handle,
                                        hipblasOperation_t   transa,
                                        hipblasOperation_t   transb,
                                        int                  m,
                                        int                  n,
                                        int                  k,
                                        const void*          alpha,
                                        const void*          A[],
                                        hipDataType          a_type,
                                        int                  lda,
                                        const void*          B[],
                                        hipDataType          b_type,
                                        int                  ldb,
                                        const void*          beta,
                                        void*                C[],
                                        hipDataType          c_type,
                                        int                  ldc,
                                        int                  batch_count,
                                        hipblasComputeType_t compute_type,
                                        hipblasGemmAlgo_t    algo)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGemmBatchedEx((cublasHandle_t)handle,
                            hipOperationToCudaOperation(transa),
                            hipOperationToCudaOperation(transb),
                            m,
                            n,
                            k,
                            alpha,
                            A,
                            HIPDatatypeToCudaDatatype_v2(a_type),
                            lda,
                            B,
                            HIPDatatypeToCudaDatatype_v2(b_type),
                            ldb,
                            beta,
                            C,
                            HIPDatatypeToCudaDatatype_v2(c_type),
                            ldc,
                            batch_count,
                            HIPComputetypeToCudaComputetype(compute_type),
                            HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmBatchedExWithFlags_v2(hipblasHandle_t      handle,
                                                 hipblasOperation_t   transa,
                                                 hipblasOperation_t   transb,
                                                 int                  m,
                                                 int                  n,
                                                 int                  k,
                                                 const void*          alpha,
                                                 const void*          A[],
                                                 hipDataType          a_type,
                                                 int                  lda,
                                                 const void*          B[],
                                                 hipDataType          b_type,
                                                 int                  ldb,
                                                 const void*          beta,
                                                 void*                C[],
                                                 hipDataType          c_type,
                                                 int                  ldc,
                                                 int                  batch_count,
                                                 hipblasComputeType_t compute_type,
                                                 hipblasGemmAlgo_t    algo,
                                                 hipblasGemmFlags_t   flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(
        cublasGemmBatchedEx((cublasHandle_t)handle,
                            hipOperationToCudaOperation(transa),
                            hipOperationToCudaOperation(transb),
                            m,
                            n,
                            k,
                            alpha,
                            A,
                            HIPDatatypeToCudaDatatype_v2(a_type),
                            lda,
                            B,
                            HIPDatatypeToCudaDatatype_v2(b_type),
                            ldb,
                            beta,
                            C,
                            HIPDatatypeToCudaDatatype_v2(c_type),
                            ldc,
                            batch_count,
                            HIPComputetypeToCudaComputetype(compute_type),
                            HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const void*        alpha,
                                            const void*        A,
                                            hipblasDatatype_t  a_type,
                                            int                lda,
                                            hipblasStride      stride_A,
                                            const void*        B,
                                            hipblasDatatype_t  b_type,
                                            int                ldb,
                                            hipblasStride      stride_B,
                                            const void*        beta,
                                            void*              C,
                                            hipblasDatatype_t  c_type,
                                            int                ldc,
                                            hipblasStride      stride_C,
                                            int                batch_count,
                                            hipblasDatatype_t  compute_type,
                                            hipblasGemmAlgo_t  algo)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGemmStridedBatchedEx((cublasHandle_t)handle,
                                   hipOperationToCudaOperation(transa),
                                   hipOperationToCudaOperation(transb),
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   A,
                                   HIPDatatypeToCudaDatatype(a_type),
                                   lda,
                                   stride_A,
                                   B,
                                   HIPDatatypeToCudaDatatype(b_type),
                                   ldb,
                                   stride_B,
                                   beta,
                                   C,
                                   HIPDatatypeToCudaDatatype(c_type),
                                   ldc,
                                   stride_C,
                                   batch_count,
                                   HIPDatatypeToCudaDatatype(compute_type),
                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmStridedBatchedExWithFlags(hipblasHandle_t    handle,
                                                     hipblasOperation_t transa,
                                                     hipblasOperation_t transb,
                                                     int                m,
                                                     int                n,
                                                     int                k,
                                                     const void*        alpha,
                                                     const void*        A,
                                                     hipblasDatatype_t  a_type,
                                                     int                lda,
                                                     hipblasStride      stride_A,
                                                     const void*        B,
                                                     hipblasDatatype_t  b_type,
                                                     int                ldb,
                                                     hipblasStride      stride_B,
                                                     const void*        beta,
                                                     void*              C,
                                                     hipblasDatatype_t  c_type,
                                                     int                ldc,
                                                     hipblasStride      stride_C,
                                                     int                batch_count,
                                                     hipblasDatatype_t  compute_type,
                                                     hipblasGemmAlgo_t  algo,
                                                     hipblasGemmFlags_t flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(
        cublasGemmStridedBatchedEx((cublasHandle_t)handle,
                                   hipOperationToCudaOperation(transa),
                                   hipOperationToCudaOperation(transb),
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   A,
                                   HIPDatatypeToCudaDatatype(a_type),
                                   lda,
                                   stride_A,
                                   B,
                                   HIPDatatypeToCudaDatatype(b_type),
                                   ldb,
                                   stride_B,
                                   beta,
                                   C,
                                   HIPDatatypeToCudaDatatype(c_type),
                                   ldc,
                                   stride_C,
                                   batch_count,
                                   HIPDatatypeToCudaDatatype(compute_type),
                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmStridedBatchedEx_v2(hipblasHandle_t      handle,
                                               hipblasOperation_t   transa,
                                               hipblasOperation_t   transb,
                                               int                  m,
                                               int                  n,
                                               int                  k,
                                               const void*          alpha,
                                               const void*          A,
                                               hipDataType          a_type,
                                               int                  lda,
                                               hipblasStride        stride_A,
                                               const void*          B,
                                               hipDataType          b_type,
                                               int                  ldb,
                                               hipblasStride        stride_B,
                                               const void*          beta,
                                               void*                C,
                                               hipDataType          c_type,
                                               int                  ldc,
                                               hipblasStride        stride_C,
                                               int                  batch_count,
                                               hipblasComputeType_t compute_type,
                                               hipblasGemmAlgo_t    algo)
try
{
    return hipCUBLASStatusToHIPStatus(
        cublasGemmStridedBatchedEx((cublasHandle_t)handle,
                                   hipOperationToCudaOperation(transa),
                                   hipOperationToCudaOperation(transb),
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   A,
                                   HIPDatatypeToCudaDatatype_v2(a_type),
                                   lda,
                                   stride_A,
                                   B,
                                   HIPDatatypeToCudaDatatype_v2(b_type),
                                   ldb,
                                   stride_B,
                                   beta,
                                   C,
                                   HIPDatatypeToCudaDatatype_v2(c_type),
                                   ldc,
                                   stride_C,
                                   batch_count,
                                   HIPComputetypeToCudaComputetype(compute_type),
                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGemmStridedBatchedExWithFlags_v2(hipblasHandle_t      handle,
                                                        hipblasOperation_t   transa,
                                                        hipblasOperation_t   transb,
                                                        int                  m,
                                                        int                  n,
                                                        int                  k,
                                                        const void*          alpha,
                                                        const void*          A,
                                                        hipDataType          a_type,
                                                        int                  lda,
                                                        hipblasStride        stride_A,
                                                        const void*          B,
                                                        hipDataType          b_type,
                                                        int                  ldb,
                                                        hipblasStride        stride_B,
                                                        const void*          beta,
                                                        void*                C,
                                                        hipDataType          c_type,
                                                        int                  ldc,
                                                        hipblasStride        stride_C,
                                                        int                  batch_count,
                                                        hipblasComputeType_t compute_type,
                                                        hipblasGemmAlgo_t    algo,
                                                        hipblasGemmFlags_t   flags)
try
{
    // flags are ignored, call original function
    return hipCUBLASStatusToHIPStatus(
        cublasGemmStridedBatchedEx((cublasHandle_t)handle,
                                   hipOperationToCudaOperation(transa),
                                   hipOperationToCudaOperation(transb),
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   A,
                                   HIPDatatypeToCudaDatatype_v2(a_type),
                                   lda,
                                   stride_A,
                                   B,
                                   HIPDatatypeToCudaDatatype_v2(b_type),
                                   ldb,
                                   stride_B,
                                   beta,
                                   C,
                                   HIPDatatypeToCudaDatatype_v2(c_type),
                                   ldc,
                                   stride_C,
                                   batch_count,
                                   HIPComputetypeToCudaComputetype(compute_type),
                                   HIPGemmAlgoToCudaGemmAlgo(algo)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsm_ex
hipblasStatus_t hipblasTrsmEx(hipblasHandle_t    handle,
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
                              hipblasDatatype_t  compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasTrsmEx_v2(hipblasHandle_t    handle,
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
                                 hipDataType        compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasTrsmBatchedEx(hipblasHandle_t    handle,
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
                                     hipblasDatatype_t  compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasTrsmBatchedEx_v2(hipblasHandle_t    handle,
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
                                        hipDataType        compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasTrsmStridedBatchedEx(hipblasHandle_t    handle,
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
                                            hipblasDatatype_t  compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasTrsmStridedBatchedEx_v2(hipblasHandle_t    handle,
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
                                               hipDataType        compute_type)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// // syrk_ex
// hipblasStatus_t hipblasCsyrkEx(hipblasHandle_t       handle,
//                                           hipblasFillMode_t     uplo,
//                                           hipblasOperation_t    trans,
//                                           int                   n,
//                                           int                   k,
//                                           const hipblasComplex* alpha,
//                                           const void*           A,
//                                           hipblasDatatype_t     Atype,
//                                           int                   lda,
//                                           const hipblasComplex* beta,
//                                           hipblasComplex*       C,
//                                           hipblasDatatype_t     Ctype,
//                                           int                   ldc)
// try
//{
//     return hipCUBLASStatusToHIPStatus(cublasCsyrkEx((cublasHandle_t)handle,
//                                                     hipFillToCudaFill(uplo),
//                                                     hipOperationToCudaOperation(trans),
//                                                     n,
//                                                     k,
//                                                     (cuComplex*)alpha,
//                                                     A,
//                                                     HIPDatatypeToCudaDatatype(Atype),
//                                                     lda,
//                                                     (cuComplex*)beta,
//                                                     (cuComplex*)C,
//                                                     HIPDatatypeToCudaDatatype(Ctype),
//                                                     ldc));
// }
// catch(...)
// {
//     return exception_to_hipblas_status();
// }

// // herk_ex
// hipblasStatus_t hipblasCherkEx(hipblasHandle_t    handle,
//                                           hipblasFillMode_t  uplo,
//                                           hipblasOperation_t trans,
//                                           int                n,
//                                           int                k,
//                                           const float*       alpha,
//                                           const void*        A,
//                                           hipblasDatatype_t  Atype,
//                                           int                lda,
//                                           const float*       beta,
//                                           hipblasComplex*    C,
//                                           hipblasDatatype_t  Ctype,
//                                           int                ldc)
// try
// {
//     return hipCUBLASStatusToHIPStatus(cublasCherkEx((cublasHandle_t)handle,
//                                                     hipFillToCudaFill(uplo),
//                                                     hipOperationToCudaOperation(trans),
//                                                     n,
//                                                     k,
//                                                     alpha,
//                                                     A,
//                                                     HIPDatatypeToCudaDatatype(Atype),
//                                                     lda,
//                                                     beta,
//                                                     (cuComplex*)C,
//                                                     HIPDatatypeToCudaDatatype(Ctype),
//                                                     ldc));
// }
// catch(...)
// {
//     return exception_to_hipblas_status();
// }

// axpy_ex
hipblasStatus_t hipblasAxpyEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       alpha,
                              hipblasDatatype_t alphaType,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              void*             y,
                              hipblasDatatype_t yType,
                              int               incy,
                              hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasAxpyEx((cublasHandle_t)handle,
                                                   n,
                                                   alpha,
                                                   HIPDatatypeToCudaDatatype(alphaType),
                                                   x,
                                                   HIPDatatypeToCudaDatatype(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToCudaDatatype(yType),
                                                   incy,
                                                   HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasAxpyEx_v2(hipblasHandle_t handle,
                                 int             n,
                                 const void*     alpha,
                                 hipDataType     alphaType,
                                 const void*     x,
                                 hipDataType     xType,
                                 int             incx,
                                 void*           y,
                                 hipDataType     yType,
                                 int             incy,
                                 hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasAxpyEx((cublasHandle_t)handle,
                                                   n,
                                                   alpha,
                                                   HIPDatatypeToCudaDatatype_v2(alphaType),
                                                   x,
                                                   HIPDatatypeToCudaDatatype_v2(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToCudaDatatype_v2(yType),
                                                   incy,
                                                   HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasAxpyBatchedEx(hipblasHandle_t   handle,
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
                                     hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasAxpyBatchedEx_v2(hipblasHandle_t handle,
                                        int             n,
                                        const void*     alpha,
                                        hipDataType     alphaType,
                                        const void*     x,
                                        hipDataType     xType,
                                        int             incx,
                                        void*           y,
                                        hipDataType     yType,
                                        int             incy,
                                        int             batch_count,
                                        hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasAxpyStridedBatchedEx(hipblasHandle_t   handle,
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
                                            hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasAxpyStridedBatchedEx_v2(hipblasHandle_t handle,
                                               int             n,
                                               const void*     alpha,
                                               hipDataType     alphaType,
                                               const void*     x,
                                               hipDataType     xType,
                                               int             incx,
                                               hipblasStride   stridex,
                                               void*           y,
                                               hipDataType     yType,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count,
                                               hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// dot_ex
hipblasStatus_t hipblasDotEx(hipblasHandle_t   handle,
                             int               n,
                             const void*       x,
                             hipblasDatatype_t xType,
                             int               incx,
                             const void*       y,
                             hipblasDatatype_t yType,
                             int               incy,
                             void*             result,
                             hipblasDatatype_t resultType,
                             hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDotEx((cublasHandle_t)handle,
                                                  n,
                                                  x,
                                                  HIPDatatypeToCudaDatatype(xType),
                                                  incx,
                                                  y,
                                                  HIPDatatypeToCudaDatatype(yType),
                                                  incy,
                                                  result,
                                                  HIPDatatypeToCudaDatatype(resultType),
                                                  HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotEx_v2(hipblasHandle_t handle,
                                int             n,
                                const void*     x,
                                hipDataType     xType,
                                int             incx,
                                const void*     y,
                                hipDataType     yType,
                                int             incy,
                                void*           result,
                                hipDataType     resultType,
                                hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDotEx((cublasHandle_t)handle,
                                                  n,
                                                  x,
                                                  HIPDatatypeToCudaDatatype_v2(xType),
                                                  incx,
                                                  y,
                                                  HIPDatatypeToCudaDatatype_v2(yType),
                                                  incy,
                                                  result,
                                                  HIPDatatypeToCudaDatatype_v2(resultType),
                                                  HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotcEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              const void*       y,
                              hipblasDatatype_t yType,
                              int               incy,
                              void*             result,
                              hipblasDatatype_t resultType,
                              hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDotcEx((cublasHandle_t)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToCudaDatatype(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToCudaDatatype(yType),
                                                   incy,
                                                   result,
                                                   HIPDatatypeToCudaDatatype(resultType),
                                                   HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotcEx_v2(hipblasHandle_t handle,
                                 int             n,
                                 const void*     x,
                                 hipDataType     xType,
                                 int             incx,
                                 const void*     y,
                                 hipDataType     yType,
                                 int             incy,
                                 void*           result,
                                 hipDataType     resultType,
                                 hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasDotcEx((cublasHandle_t)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToCudaDatatype_v2(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToCudaDatatype_v2(yType),
                                                   incy,
                                                   result,
                                                   HIPDatatypeToCudaDatatype_v2(resultType),
                                                   HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDotBatchedEx(hipblasHandle_t   handle,
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
                                    hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotBatchedEx_v2(hipblasHandle_t handle,
                                       int             n,
                                       const void*     x,
                                       hipDataType     xType,
                                       int             incx,
                                       const void*     y,
                                       hipDataType     yType,
                                       int             incy,
                                       int             batch_count,
                                       void*           result,
                                       hipDataType     resultType,
                                       hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotcBatchedEx(hipblasHandle_t   handle,
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
                                     hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotcBatchedEx_v2(hipblasHandle_t handle,
                                        int             n,
                                        const void*     x,
                                        hipDataType     xType,
                                        int             incx,
                                        const void*     y,
                                        hipDataType     yType,
                                        int             incy,
                                        int             batch_count,
                                        void*           result,
                                        hipDataType     resultType,
                                        hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotStridedBatchedEx(hipblasHandle_t   handle,
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
                                           hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotStridedBatchedEx_v2(hipblasHandle_t handle,
                                              int             n,
                                              const void*     x,
                                              hipDataType     xType,
                                              int             incx,
                                              hipblasStride   stridex,
                                              const void*     y,
                                              hipDataType     yType,
                                              int             incy,
                                              hipblasStride   stridey,
                                              int             batch_count,
                                              void*           result,
                                              hipDataType     resultType,
                                              hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotcStridedBatchedEx(hipblasHandle_t   handle,
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
                                            hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDotcStridedBatchedEx_v2(hipblasHandle_t handle,
                                               int             n,
                                               const void*     x,
                                               hipDataType     xType,
                                               int             incx,
                                               hipblasStride   stridex,
                                               const void*     y,
                                               hipDataType     yType,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count,
                                               void*           result,
                                               hipDataType     resultType,
                                               hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// nrm2_ex
hipblasStatus_t hipblasNrm2Ex(hipblasHandle_t   handle,
                              int               n,
                              const void*       x,
                              hipblasDatatype_t xType,
                              int               incx,
                              void*             result,
                              hipblasDatatype_t resultType,
                              hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasNrm2Ex((cublasHandle_t)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToCudaDatatype(xType),
                                                   incx,
                                                   result,
                                                   HIPDatatypeToCudaDatatype(resultType),
                                                   HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasNrm2Ex_v2(hipblasHandle_t handle,
                                 int             n,
                                 const void*     x,
                                 hipDataType     xType,
                                 int             incx,
                                 void*           result,
                                 hipDataType     resultType,
                                 hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasNrm2Ex((cublasHandle_t)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToCudaDatatype_v2(xType),
                                                   incx,
                                                   result,
                                                   HIPDatatypeToCudaDatatype_v2(resultType),
                                                   HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasNrm2BatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     int               batch_count,
                                     void*             result,
                                     hipblasDatatype_t resultType,
                                     hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasNrm2BatchedEx_v2(hipblasHandle_t handle,
                                        int             n,
                                        const void*     x,
                                        hipDataType     xType,
                                        int             incx,
                                        int             batch_count,
                                        void*           result,
                                        hipDataType     resultType,
                                        hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasNrm2StridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            int               batch_count,
                                            void*             result,
                                            hipblasDatatype_t resultType,
                                            hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasNrm2StridedBatchedEx_v2(hipblasHandle_t handle,
                                               int             n,
                                               const void*     x,
                                               hipDataType     xType,
                                               int             incx,
                                               hipblasStride   stridex,
                                               int             batch_count,
                                               void*           result,
                                               hipDataType     resultType,
                                               hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// rot_ex
hipblasStatus_t hipblasRotEx(hipblasHandle_t   handle,
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
                             hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasRotEx((cublasHandle_t)handle,
                                                  n,
                                                  x,
                                                  HIPDatatypeToCudaDatatype(xType),
                                                  incx,
                                                  y,
                                                  HIPDatatypeToCudaDatatype(yType),
                                                  incy,
                                                  c,
                                                  s,
                                                  HIPDatatypeToCudaDatatype(csType),
                                                  HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasRotEx_v2(hipblasHandle_t handle,
                                int             n,
                                void*           x,
                                hipDataType     xType,
                                int             incx,
                                void*           y,
                                hipDataType     yType,
                                int             incy,
                                const void*     c,
                                const void*     s,
                                hipDataType     csType,
                                hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasRotEx((cublasHandle_t)handle,
                                                  n,
                                                  x,
                                                  HIPDatatypeToCudaDatatype_v2(xType),
                                                  incx,
                                                  y,
                                                  HIPDatatypeToCudaDatatype_v2(yType),
                                                  incy,
                                                  c,
                                                  s,
                                                  HIPDatatypeToCudaDatatype_v2(csType),
                                                  HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasRotBatchedEx(hipblasHandle_t   handle,
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
                                    hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasRotBatchedEx_v2(hipblasHandle_t handle,
                                       int             n,
                                       void*           x,
                                       hipDataType     xType,
                                       int             incx,
                                       void*           y,
                                       hipDataType     yType,
                                       int             incy,
                                       const void*     c,
                                       const void*     s,
                                       hipDataType     csType,
                                       int             batch_count,
                                       hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasRotStridedBatchedEx(hipblasHandle_t   handle,
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
                                           hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasRotStridedBatchedEx_v2(hipblasHandle_t handle,
                                              int             n,
                                              void*           x,
                                              hipDataType     xType,
                                              int             incx,
                                              hipblasStride   stridex,
                                              void*           y,
                                              hipDataType     yType,
                                              int             incy,
                                              hipblasStride   stridey,
                                              const void*     c,
                                              const void*     s,
                                              hipDataType     csType,
                                              int             batch_count,
                                              hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

// scal_ex
hipblasStatus_t hipblasScalEx(hipblasHandle_t   handle,
                              int               n,
                              const void*       alpha,
                              hipblasDatatype_t alphaType,
                              void*             x,
                              hipblasDatatype_t xType,
                              int               incx,
                              hipblasDatatype_t executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasScalEx((cublasHandle_t)handle,
                                                   n,
                                                   alpha,
                                                   HIPDatatypeToCudaDatatype(alphaType),
                                                   x,
                                                   HIPDatatypeToCudaDatatype(xType),
                                                   incx,
                                                   HIPDatatypeToCudaDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScalEx_v2(hipblasHandle_t handle,
                                 int             n,
                                 const void*     alpha,
                                 hipDataType     alphaType,
                                 void*           x,
                                 hipDataType     xType,
                                 int             incx,
                                 hipDataType     executionType)
try
{
    return hipCUBLASStatusToHIPStatus(cublasScalEx((cublasHandle_t)handle,
                                                   n,
                                                   alpha,
                                                   HIPDatatypeToCudaDatatype_v2(alphaType),
                                                   x,
                                                   HIPDatatypeToCudaDatatype_v2(xType),
                                                   incx,
                                                   HIPDatatypeToCudaDatatype_v2(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScalBatchedEx(hipblasHandle_t   handle,
                                     int               n,
                                     const void*       alpha,
                                     hipblasDatatype_t alphaType,
                                     void*             x,
                                     hipblasDatatype_t xType,
                                     int               incx,
                                     int               batch_count,
                                     hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScalBatchedEx_v2(hipblasHandle_t handle,
                                        int             n,
                                        const void*     alpha,
                                        hipDataType     alphaType,
                                        void*           x,
                                        hipDataType     xType,
                                        int             incx,
                                        int             batch_count,
                                        hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScalStridedBatchedEx(hipblasHandle_t   handle,
                                            int               n,
                                            const void*       alpha,
                                            hipblasDatatype_t alphaType,
                                            void*             x,
                                            hipblasDatatype_t xType,
                                            int               incx,
                                            hipblasStride     stridex,
                                            int               batch_count,
                                            hipblasDatatype_t executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasScalStridedBatchedEx_v2(hipblasHandle_t handle,
                                               int             n,
                                               const void*     alpha,
                                               hipDataType     alphaType,
                                               void*           x,
                                               hipDataType     xType,
                                               int             incx,
                                               hipblasStride   stridex,
                                               int             batch_count,
                                               hipDataType     executionType)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

#ifdef __cplusplus
}

#endif
