/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "hipblas/hipblas.h"
#include "exceptions.hpp"
#include "limits.h"
#include "rocblas/rocblas.h"
#ifdef __HIP_PLATFORM_SOLVER__
#include "rocsolver/rocsolver.h"
#endif
#include <algorithm>
#include <functional>
#include <math.h>

extern "C" hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status_ error);

// Attempt a rocBLAS call; if it gets an allocation error, query the
// size needed and attempt to allocate it, retrying the operation
static hipblasStatus_t hipblasDemandAlloc(rocblas_handle                   handle,
                                          std::function<hipblasStatus_t()> func)
{
    hipblasStatus_t status = func();
    if(status == HIPBLAS_STATUS_ALLOC_FAILED)
    {
        rocblas_status blas_status = rocblas_start_device_memory_size_query(handle);
        if(blas_status != rocblas_status_success)
            status = rocBLASStatusToHIPStatus(blas_status);
        else
        {
            status = func();
            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                size_t size;
                blas_status = rocblas_stop_device_memory_size_query(handle, &size);
                if(blas_status != rocblas_status_success)
                    status = rocBLASStatusToHIPStatus(blas_status);
                else
                {
                    blas_status = rocblas_set_device_memory_size(handle, size);
                    if(blas_status != rocblas_status_success)
                        status = rocBLASStatusToHIPStatus(blas_status);
                    else
                        status = func();
                }
            }
        }
    }
    return status;
}

#define HIPBLAS_DEMAND_ALLOC(status__) \
    hipblasDemandAlloc(rocblas_handle(handle), [&]() -> hipblasStatus_t { return status__; })

extern "C" {

rocblas_operation_ hipOperationToHCCOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasOperation_t HCCOperationToHIPOperation(rocblas_operation_ op)
{
    switch(op)
    {
    case rocblas_operation_none:
        return HIPBLAS_OP_N;
    case rocblas_operation_transpose:
        return HIPBLAS_OP_T;
    case rocblas_operation_conjugate_transpose:
        return HIPBLAS_OP_C;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_fill_ hipFillToHCCFill(hipblasFillMode_t fill)
{
    switch(fill)
    {
    case HIPBLAS_FILL_MODE_UPPER:
        return rocblas_fill_upper;
    case HIPBLAS_FILL_MODE_LOWER:
        return rocblas_fill_lower;
    case HIPBLAS_FILL_MODE_FULL:
        return rocblas_fill_full;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasFillMode_t HCCFillToHIPFill(rocblas_fill_ fill)
{
    switch(fill)
    {
    case rocblas_fill_upper:
        return HIPBLAS_FILL_MODE_UPPER;
    case rocblas_fill_lower:
        return HIPBLAS_FILL_MODE_LOWER;
    case rocblas_fill_full:
        return HIPBLAS_FILL_MODE_FULL;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_diagonal_ hipDiagonalToHCCDiagonal(hipblasDiagType_t diagonal)
{
    switch(diagonal)
    {
    case HIPBLAS_DIAG_NON_UNIT:
        return rocblas_diagonal_non_unit;
    case HIPBLAS_DIAG_UNIT:
        return rocblas_diagonal_unit;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasDiagType_t HCCDiagonalToHIPDiagonal(rocblas_diagonal_ diagonal)
{
    switch(diagonal)
    {
    case rocblas_diagonal_non_unit:
        return HIPBLAS_DIAG_NON_UNIT;
    case rocblas_diagonal_unit:
        return HIPBLAS_DIAG_UNIT;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_side_ hipSideToHCCSide(hipblasSideMode_t side)
{
    switch(side)
    {
    case HIPBLAS_SIDE_LEFT:
        return rocblas_side_left;
    case HIPBLAS_SIDE_RIGHT:
        return rocblas_side_right;
    case HIPBLAS_SIDE_BOTH:
        return rocblas_side_both;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasSideMode_t HCCSideToHIPSide(rocblas_side_ side)
{
    switch(side)
    {
    case rocblas_side_left:
        return HIPBLAS_SIDE_LEFT;
    case rocblas_side_right:
        return HIPBLAS_SIDE_RIGHT;
    case rocblas_side_both:
        return HIPBLAS_SIDE_BOTH;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_pointer_mode HIPPointerModeToRocblasPointerMode(hipblasPointerMode_t mode)
{
    switch(mode)
    {
    case HIPBLAS_POINTER_MODE_HOST:
        return rocblas_pointer_mode_host;

    case HIPBLAS_POINTER_MODE_DEVICE:
        return rocblas_pointer_mode_device;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasPointerMode_t RocblasPointerModeToHIPPointerMode(rocblas_pointer_mode mode)
{
    switch(mode)
    {
    case rocblas_pointer_mode_host:
        return HIPBLAS_POINTER_MODE_HOST;

    case rocblas_pointer_mode_device:
        return HIPBLAS_POINTER_MODE_DEVICE;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_datatype HIPDatatypeToRocblasDatatype(hipblasDatatype_t type)
{
    switch(type)
    {
    case HIPBLAS_R_16B:
        return rocblas_datatype_bf16_r;

    case HIPBLAS_R_16F:
        return rocblas_datatype_f16_r;

    case HIPBLAS_R_32F:
        return rocblas_datatype_f32_r;

    case HIPBLAS_R_64F:
        return rocblas_datatype_f64_r;

    case HIPBLAS_R_8I:
        return rocblas_datatype_i8_r;

    case HIPBLAS_R_32I:
        return rocblas_datatype_i32_r;

    case HIPBLAS_C_16F:
        return rocblas_datatype_f16_c;

    case HIPBLAS_C_32F:
        return rocblas_datatype_f32_c;

    case HIPBLAS_C_64F:
        return rocblas_datatype_f64_c;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasDatatype_t RocblasDatatypeToHIPDatatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return HIPBLAS_R_16F;

    case rocblas_datatype_f32_r:
        return HIPBLAS_R_32F;

    case rocblas_datatype_f64_r:
        return HIPBLAS_R_64F;

    case rocblas_datatype_i8_r:
        return HIPBLAS_R_8I;

    case rocblas_datatype_i32_r:
        return HIPBLAS_R_32I;

    case rocblas_datatype_f16_c:
        return HIPBLAS_C_16F;

    case rocblas_datatype_f32_c:
        return HIPBLAS_C_32F;

    case rocblas_datatype_f64_c:
        return HIPBLAS_C_64F;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_gemm_algo HIPGemmAlgoToRocblasGemmAlgo(hipblasGemmAlgo_t algo)
{
    switch(algo)
    {
    case HIPBLAS_GEMM_DEFAULT:
        return rocblas_gemm_algo_standard;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasGemmAlgo_t RocblasGemmAlgoToHIPGemmAlgo(rocblas_gemm_algo algo)
{
    switch(algo)
    {
    case rocblas_gemm_algo_standard:
        return HIPBLAS_GEMM_DEFAULT;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

rocblas_atomics_mode HIPAtomicsModeToRocblasAtomicsMode(hipblasAtomicsMode_t mode)
{
    switch(mode)
    {
    case HIPBLAS_ATOMICS_NOT_ALLOWED:
        return rocblas_atomics_not_allowed;
    case HIPBLAS_ATOMICS_ALLOWED:
        return rocblas_atomics_allowed;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasAtomicsMode_t RocblasAtomicsModeToHIPAtomicsMode(rocblas_atomics_mode mode)
{
    switch(mode)
    {
    case rocblas_atomics_not_allowed:
        return HIPBLAS_ATOMICS_NOT_ALLOWED;
    case rocblas_atomics_allowed:
        return HIPBLAS_ATOMICS_ALLOWED;
    }
    throw HIPBLAS_STATUS_INVALID_ENUM;
}

hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status_ error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    default:
        return HIPBLAS_STATUS_UNKNOWN;
    }
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
try
{
    if(!handle)
        return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;

    // Create the rocBLAS handle
    return rocBLASStatusToHIPStatus(rocblas_create_handle((rocblas_handle*)handle));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
try
{
    return rocBLASStatusToHIPStatus(rocblas_destroy_handle((rocblas_handle)handle));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return rocBLASStatusToHIPStatus(rocblas_set_stream((rocblas_handle)handle, streamId));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return rocBLASStatusToHIPStatus(rocblas_get_stream((rocblas_handle)handle, streamId));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_pointer_mode((rocblas_handle)handle, HIPPointerModeToRocblasPointerMode(mode)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
try
{
    rocblas_pointer_mode rocblas_mode;
    rocblas_status       status = rocblas_get_pointer_mode((rocblas_handle)handle, &rocblas_mode);
    *mode                       = RocblasPointerModeToHIPPointerMode(rocblas_mode);
    return rocBLASStatusToHIPStatus(status);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_set_vector(n, elemSize, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_get_vector(n, elemSize, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_set_matrix(rows, cols, elemSize, A, lda, B, ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_get_matrix(rows, cols, elemSize, A, lda, B, ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_vector_async(n, elemSize, x, incx, y, incy, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_vector_async(n, elemSize, x, incx, y, incy, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_matrix_async(rows, cols, elemSize, A, lda, B, ldb, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_matrix_async(rows, cols, elemSize, A, lda, B, ldb, stream));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// atomics mode
hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode)
try
{
    return rocBLASStatusToHIPStatus(rocblas_set_atomics_mode(
        (rocblas_handle)handle, HIPAtomicsModeToRocblasAtomicsMode(atomics_mode)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_atomics_mode((rocblas_handle)handle, (rocblas_atomics_mode*)atomics_mode));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_isamax((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_idamax((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamax((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamax((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax_batched
hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_isamax_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_idamax_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     int*                        result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_icamax_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     int*                              result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_izamax_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amax_strided_batched
hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batch_count,
                                            int*            result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_isamax_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batch_count,
                                            int*            result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_idamax_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batch_count,
                                            int*                  result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_icamax_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batch_count,
                                            int*                        result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_izamax_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amin
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_isamin((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_idamin((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamin((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamin(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamin((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amin_batched
hipblasStatus_t hipblasIsaminBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_isamin_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdaminBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_idamin_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     int*                        result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_icamin_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     int*                              result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_izamin_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// amin_strided_batched
hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batch_count,
                                            int*            result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_isamin_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batch_count,
                                            int*            result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_idamin_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batch_count,
                                            int*                  result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_icamin_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batch_count,
                                            int*                        result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_izamin_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// asum
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sasum((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dasum((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_scasum((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dzasum((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, float* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_sasum_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batch_count,
                                    double*             result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dasum_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     float*                      result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     double*                           result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dzasum_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batch_count,
                                           float*          result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sasum_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batch_count,
                                           double*         result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dasum_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batch_count,
                                            float*                result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batch_count,
                                            double*                     result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dzasum_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// axpy
hipblasStatus_t hipblasHaxpy(hipblasHandle_t    handle,
                             int                n,
                             const hipblasHalf* alpha,
                             const hipblasHalf* x,
                             int                incx,
                             hipblasHalf*       y,
                             int                incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_haxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_half*)alpha,
                                                  (rocblas_half*)x,
                                                  incx,
                                                  (rocblas_half*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSaxpy(
    hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_saxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
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
    return rocBLASStatusToHIPStatus(
        rocblas_daxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
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
    return rocBLASStatusToHIPStatus(rocblas_caxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
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
    return rocBLASStatusToHIPStatus(rocblas_zaxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
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
                                    int                      batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_haxpy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_half*)alpha,
                                                          (rocblas_half* const*)x,
                                                          incx,
                                                          (rocblas_half* const*)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_saxpy_batched((rocblas_handle)handle, n, alpha, x, incx, y, incy, batch_count));
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
                                    int                 batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_daxpy_batched((rocblas_handle)handle, n, alpha, x, incx, y, incy, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_caxpy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex* const*)x,
                                                          incx,
                                                          (rocblas_float_complex* const*)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zaxpy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex* const*)x,
                                                          incx,
                                                          (rocblas_double_complex* const*)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           int                batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_haxpy_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_half*)alpha,
                                                                  (rocblas_half*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_half*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           int             batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_saxpy_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           int             batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_daxpy_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           int                   batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_caxpy_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           int                         batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zaxpy_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// copy
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_scopy((rocblas_handle)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dcopy((rocblas_handle)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopy(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
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
    return rocBLASStatusToHIPStatus(rocblas_zcopy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_scopy_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dcopy_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zcopy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_scopy_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dcopy_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zcopy_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// dot
hipblasStatus_t hipblasHdot(hipblasHandle_t    handle,
                            int                n,
                            const hipblasHalf* x,
                            int                incx,
                            const hipblasHalf* y,
                            int                incy,
                            hipblasHalf*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_hdot((rocblas_handle)handle,
                                                 n,
                                                 (rocblas_half*)x,
                                                 incx,
                                                 (rocblas_half*)y,
                                                 incy,
                                                 (rocblas_half*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasBfdot(hipblasHandle_t        handle,
                             int                    n,
                             const hipblasBfloat16* x,
                             int                    incx,
                             const hipblasBfloat16* y,
                             int                    incy,
                             hipblasBfloat16*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_bfdot((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_bfloat16*)x,
                                                  incx,
                                                  (rocblas_bfloat16*)y,
                                                  incy,
                                                  (rocblas_bfloat16*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_sdot((rocblas_handle)handle, n, x, incx, y, incy, result));
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
    return rocBLASStatusToHIPStatus(
        rocblas_ddot((rocblas_handle)handle, n, x, incx, y, incy, result));
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
    return rocBLASStatusToHIPStatus(rocblas_cdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
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
    return rocBLASStatusToHIPStatus(rocblas_cdotu((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
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
    return rocBLASStatusToHIPStatus(rocblas_zdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)result));
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
    return rocBLASStatusToHIPStatus(rocblas_zdotu((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)result));
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
                                   int                      batch_count,
                                   hipblasHalf*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_hdot_batched((rocblas_handle)handle,
                                                         n,
                                                         (rocblas_half* const*)x,
                                                         incx,
                                                         (rocblas_half* const*)y,
                                                         incy,
                                                         batch_count,
                                                         (rocblas_half*)result));
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
                                    int                          batch_count,
                                    hipblasBfloat16*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_bfdot_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_bfloat16* const*)x,
                                                          incx,
                                                          (rocblas_bfloat16* const*)y,
                                                          incy,
                                                          batch_count,
                                                          (rocblas_bfloat16*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batch_count,
                                   float*             result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_sdot_batched((rocblas_handle)handle, n, x, incx, y, incy, batch_count, result));
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
                                   int                 batch_count,
                                   double*             result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ddot_batched((rocblas_handle)handle, n, x, incx, y, incy, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batch_count,
                                    hipblasComplex*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdotc_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batch_count,
                                                          (rocblas_float_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batch_count,
                                    hipblasComplex*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdotu_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batch_count,
                                                          (rocblas_float_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batch_count,
                                    hipblasDoubleComplex*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdotc_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batch_count,
                                                          (rocblas_double_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batch_count,
                                    hipblasDoubleComplex*             result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdotu_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batch_count,
                                                          (rocblas_double_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                          int                batch_count,
                                          hipblasHalf*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_hdot_strided_batched((rocblas_handle)handle,
                                                                 n,
                                                                 (rocblas_half*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_half*)y,
                                                                 incy,
                                                                 stridey,
                                                                 batch_count,
                                                                 (rocblas_half*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                           int                    n,
                                           const hipblasBfloat16* x,
                                           int                    incx,
                                           hipblasStride          stridex,
                                           const hipblasBfloat16* y,
                                           int                    incy,
                                           hipblasStride          stridey,
                                           int                    batch_count,
                                           hipblasBfloat16*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_bfdot_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_bfloat16*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_bfloat16*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count,
                                                                  (rocblas_bfloat16*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const float*    y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batch_count,
                                          float*          result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sdot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          hipblasStride   stridex,
                                          const double*   y,
                                          int             incy,
                                          hipblasStride   stridey,
                                          int             batch_count,
                                          double*         result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ddot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batch_count, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batch_count,
                                           hipblasComplex*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdotc_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count,
                                                                  (rocblas_float_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           hipblasStride         stridey,
                                           int                   batch_count,
                                           hipblasComplex*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdotu_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count,
                                                                  (rocblas_float_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batch_count,
                                           hipblasDoubleComplex*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdotc_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count,
                                                                  (rocblas_double_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           hipblasStride               stridey,
                                           int                         batch_count,
                                           hipblasDoubleComplex*       result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdotu_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batch_count,
                                                                  (rocblas_double_complex*)result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2
hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_snrm2((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dnrm2((rocblas_handle)handle, n, x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_scnrm2((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dznrm2((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_snrm2_batched((rocblas_handle)handle, n, x, incx, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dnrm2_batched((rocblas_handle)handle, n, x, incx, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dznrm2_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           float*          result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_snrm2_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount,
                                           double*         result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dnrm2_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount,
                                            float*                result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            hipblasStride               stridex,
                                            int                         batchCount,
                                            double*                     result)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dznrm2_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batchCount, result));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_srot((rocblas_handle)handle, n, x, incx, y, incy, c, s));
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
    return rocBLASStatusToHIPStatus(
        rocblas_drot((rocblas_handle)handle, n, x, incx, y, incy, c, s));
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
    return rocBLASStatusToHIPStatus(rocblas_crot((rocblas_handle)handle,
                                                 n,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)y,
                                                 incy,
                                                 c,
                                                 (rocblas_float_complex*)s));
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
    return rocBLASStatusToHIPStatus(rocblas_csrot((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  c,
                                                  s));
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
    return rocBLASStatusToHIPStatus(rocblas_zrot((rocblas_handle)handle,
                                                 n,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)y,
                                                 incy,
                                                 c,
                                                 (rocblas_double_complex*)s));
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
    return rocBLASStatusToHIPStatus(rocblas_zdrot((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  c,
                                                  s));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_srot_batched((rocblas_handle)handle, n, x, incx, y, incy, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_drot_batched((rocblas_handle)handle, n, x, incx, y, incy, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_crot_batched((rocblas_handle)handle,
                                                         n,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)y,
                                                         incy,
                                                         c,
                                                         (rocblas_float_complex*)s,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csrot_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          c,
                                                          s,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zrot_batched((rocblas_handle)handle,
                                                         n,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)y,
                                                         incy,
                                                         c,
                                                         (rocblas_double_complex*)s,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdrot_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          c,
                                                          s,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_srot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_drot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_crot_strided_batched((rocblas_handle)handle,
                                                                 n,
                                                                 (rocblas_float_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_float_complex*)y,
                                                                 incy,
                                                                 stridey,
                                                                 c,
                                                                 (rocblas_float_complex*)s,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csrot_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  c,
                                                                  s,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zrot_strided_batched((rocblas_handle)handle,
                                                                 n,
                                                                 (rocblas_double_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_double_complex*)y,
                                                                 incy,
                                                                 stridey,
                                                                 c,
                                                                 (rocblas_double_complex*)s,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdrot_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  c,
                                                                  s,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotg
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
try
{
    return rocBLASStatusToHIPStatus(rocblas_srotg((rocblas_handle)handle, a, b, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s)
try
{
    return rocBLASStatusToHIPStatus(rocblas_drotg((rocblas_handle)handle, a, b, c, s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotg(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
try
{
    return rocBLASStatusToHIPStatus(rocblas_crotg((rocblas_handle)handle,
                                                  (rocblas_float_complex*)a,
                                                  (rocblas_float_complex*)b,
                                                  c,
                                                  (rocblas_float_complex*)s));
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
    return rocBLASStatusToHIPStatus(rocblas_zrotg((rocblas_handle)handle,
                                                  (rocblas_double_complex*)a,
                                                  (rocblas_double_complex*)b,
                                                  c,
                                                  (rocblas_double_complex*)s));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotg_batched
hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                    float* const    a[],
                                    float* const    b[],
                                    float* const    c[],
                                    float* const    s[],
                                    int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotg_batched((rocblas_handle)handle, a, b, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                    double* const   a[],
                                    double* const   b[],
                                    double* const   c[],
                                    double* const   s[],
                                    int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotg_batched((rocblas_handle)handle, a, b, c, s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                    hipblasComplex* const a[],
                                    hipblasComplex* const b[],
                                    float* const          c[],
                                    hipblasComplex* const s[],
                                    int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_crotg_batched((rocblas_handle)handle,
                                                          (rocblas_float_complex**)a,
                                                          (rocblas_float_complex**)b,
                                                          c,
                                                          (rocblas_float_complex**)s,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                    hipblasDoubleComplex* const a[],
                                    hipblasDoubleComplex* const b[],
                                    double* const               c[],
                                    hipblasDoubleComplex* const s[],
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zrotg_batched((rocblas_handle)handle,
                                                          (rocblas_double_complex**)a,
                                                          (rocblas_double_complex**)b,
                                                          c,
                                                          (rocblas_double_complex**)s,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_srotg_strided_batched(
        (rocblas_handle)handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_drotg_strided_batched(
        (rocblas_handle)handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_crotg_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_float_complex*)a,
                                                                  stride_a,
                                                                  (rocblas_float_complex*)b,
                                                                  stride_b,
                                                                  c,
                                                                  stride_c,
                                                                  (rocblas_float_complex*)s,
                                                                  stride_s,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zrotg_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_double_complex*)a,
                                                                  stride_a,
                                                                  (rocblas_double_complex*)b,
                                                                  stride_b,
                                                                  c,
                                                                  stride_c,
                                                                  (rocblas_double_complex*)s,
                                                                  stride_s,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotm
hipblasStatus_t hipblasSrotm(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotm((rocblas_handle)handle, n, x, incx, y, incy, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotm(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotm((rocblas_handle)handle, n, x, incx, y, incy, param));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotm_batched((rocblas_handle)handle, n, x, incx, y, incy, param, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    double* const       x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    const double* const param[],
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotm_batched((rocblas_handle)handle, n, x, incx, y, incy, param, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           hipblasStride   strideparam,
                                           int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_srotm_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  param,
                                                                  strideparam,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           hipblasStride   strideparam,
                                           int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_drotm_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  param,
                                                                  strideparam,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// rotmg
hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
try
{
    return rocBLASStatusToHIPStatus(rocblas_srotmg((rocblas_handle)handle, d1, d2, x1, y1, param));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param)
try
{
    return rocBLASStatusToHIPStatus(rocblas_drotmg((rocblas_handle)handle, d1, d2, x1, y1, param));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotmg_batched((rocblas_handle)handle, d1, d2, x1, y1, param, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                     double* const       d1[],
                                     double* const       d2[],
                                     double* const       x1[],
                                     const double* const y1[],
                                     double* const       param[],
                                     int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotmg_batched((rocblas_handle)handle, d1, d2, x1, y1, param, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                            hipblasStride   strideparam,
                                            int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_srotmg_strided_batched((rocblas_handle)handle,
                                                                   d1,
                                                                   stride_d1,
                                                                   d2,
                                                                   stride_d2,
                                                                   x1,
                                                                   stride_x1,
                                                                   y1,
                                                                   stride_y1,
                                                                   param,
                                                                   strideparam,
                                                                   batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                            hipblasStride   strideparam,
                                            int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_drotmg_strided_batched((rocblas_handle)handle,
                                                                   d1,
                                                                   stride_d1,
                                                                   d2,
                                                                   stride_d2,
                                                                   x1,
                                                                   stride_x1,
                                                                   y1,
                                                                   stride_y1,
                                                                   param,
                                                                   strideparam,
                                                                   batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// scal
hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sscal((rocblas_handle)handle, n, alpha, x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dscal((rocblas_handle)handle, n, alpha, x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscal(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cscal(
        (rocblas_handle)handle, n, (rocblas_float_complex*)alpha, (rocblas_float_complex*)x, incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_csscal((rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx));
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
    return rocBLASStatusToHIPStatus(rocblas_zscal((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscal(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zdscal((rocblas_handle)handle, n, alpha, (rocblas_double_complex*)x, incx));
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
    return rocBLASStatusToHIPStatus(
        rocblas_sscal_batched((rocblas_handle)handle, n, alpha, x, incx, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dscal_batched((rocblas_handle)handle, n, alpha, x, incx, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex* const*)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex* const*)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                     int                   n,
                                     const float*          alpha,
                                     hipblasComplex* const x[],
                                     int                   incx,
                                     int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex* const*)x, incx, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const double*               alpha,
                                     hipblasDoubleComplex* const x[],
                                     int                         incx,
                                     int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdscal_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_double_complex* const*)x, incx, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           float*          x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sscal_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double*         x,
                                           int             incx,
                                           hipblasStride   stridex,
                                           int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dscal_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zscal_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            hipblasComplex* x,
                                            int             incx,
                                            hipblasStride   stridex,
                                            int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_strided_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx, stridex, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const double*         alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            hipblasStride         stridex,
                                            int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdscal_strided_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_double_complex*)x, incx, stridex, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// swap
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sswap((rocblas_handle)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dswap((rocblas_handle)handle, n, x, incx, y, incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswap(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cswap((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
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
    return rocBLASStatusToHIPStatus(rocblas_zswap((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// swap_batched
hipblasStatus_t hipblasSswapBatched(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_sswap_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDswapBatched(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dswap_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    hipblasComplex* x[],
                                    int             incx,
                                    hipblasComplex* y[],
                                    int             incy,
                                    int             batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cswap_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x[],
                                    int                   incx,
                                    hipblasDoubleComplex* y[],
                                    int                   incy,
                                    int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zswap_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sswap_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dswap_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cswap_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zswap_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sgbmv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
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
    return rocBLASStatusToHIPStatus(rocblas_dgbmv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
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
    return rocBLASStatusToHIPStatus(rocblas_cgbmv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zgbmv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  m,
                                                  n,
                                                  kl,
                                                  ku,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sgbmv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
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
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dgbmv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
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
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgbmv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          kl,
                                                          ku,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgbmv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          kl,
                                                          ku,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sgbmv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  kl,
                                                                  ku,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  stride_a,
                                                                  x,
                                                                  incx,
                                                                  stride_x,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dgbmv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  kl,
                                                                  ku,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  stride_a,
                                                                  x,
                                                                  incx,
                                                                  stride_x,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgbmv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  kl,
                                                                  ku,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  stride_a,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgbmv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  kl,
                                                                  ku,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  stride_a,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sgemv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
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
    return rocBLASStatusToHIPStatus(rocblas_dgemv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
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
    return rocBLASStatusToHIPStatus(rocblas_cgemv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zgemv((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_sgemv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dgemv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgemv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgemv_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sgemv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dgemv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgemv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgemv_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_sger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(rocblas_cgeru((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_cgerc((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zgeru((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zgerc((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)A,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sger_batched(
        (rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dger_batched(
        (rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgeru_batched((rocblas_handle)handle,
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgerc_batched((rocblas_handle)handle,
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgeru_batched((rocblas_handle)handle,
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgerc_batched((rocblas_handle)handle,
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sger_strided_batched((rocblas_handle)handle,
                                                                 m,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 y,
                                                                 incy,
                                                                 stridey,
                                                                 A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dger_strided_batched((rocblas_handle)handle,
                                                                 m,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 y,
                                                                 incy,
                                                                 stridey,
                                                                 A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgeru_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgerc_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgeru_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgerc_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  k,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zhbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  k,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          k,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          k,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chbmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  k,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhbmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  k,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chemv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zhemv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chemv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhemv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chemv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  stride_a,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhemv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  stride_a,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stride_y,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_cher((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zher((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)A,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cher_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)A,
                                                         lda,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zher_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)A,
                                                         lda,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cher_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 (rocblas_float_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_float_complex*)A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zher_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 (rocblas_double_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_double_complex*)A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_cher2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zher2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)A,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cher2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zher2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cher2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zher2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)AP,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zhpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)AP,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)AP,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)AP,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)AP,
                                                                  strideAP,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)AP,
                                                                  strideAP,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chpr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)AP));
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
    return rocBLASStatusToHIPStatus(rocblas_zhpr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)AP));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)AP,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const double*                     alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)AP,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 (rocblas_float_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_float_complex*)AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 (rocblas_double_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_double_complex*)AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chpr2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)AP));
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
    return rocBLASStatusToHIPStatus(rocblas_zhpr2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)AP));
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
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    hipblasComplex* const       AP[],
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          (rocblas_float_complex**)AP,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZhpr2Batched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    hipblasDoubleComplex* const       AP[],
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          (rocblas_double_complex**)AP,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chpr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_float_complex*)AP,
                                                                  strideAP,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhpr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_double_complex*)AP,
                                                                  strideAP,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssbmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
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
    return rocBLASStatusToHIPStatus(rocblas_dsbmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
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
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssbmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  k,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsbmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  k,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sspmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, AP, x, incx, beta, y, incy));
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
    return rocBLASStatusToHIPStatus(rocblas_dspmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, AP, x, incx, beta, y, incy));
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
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          AP,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDspmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const AP[],
                                    const double* const x[],
                                    int                 incx,
                                    const double*       beta,
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          AP,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  AP,
                                                                  strideAP,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspmv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  AP,
                                                                  strideAP,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_sspr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dspr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cspr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_float_complex*)alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zspr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_double_complex*)alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)AP));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDsprBatched(hipblasHandle_t     handle,
                                   hipblasFillMode_t   uplo,
                                   int                 n,
                                   const double*       alpha,
                                   const double* const x[],
                                   int                 incx,
                                   double* const       AP[],
                                   int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCsprBatched(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int                         n,
                                   const hipblasComplex*       alpha,
                                   const hipblasComplex* const x[],
                                   int                         incx,
                                   hipblasComplex* const       AP[],
                                   int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_cspr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_float_complex*)alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)AP,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZsprBatched(hipblasHandle_t                   handle,
                                   hipblasFillMode_t                 uplo,
                                   int                               n,
                                   const hipblasDoubleComplex*       alpha,
                                   const hipblasDoubleComplex* const x[],
                                   int                               incx,
                                   hipblasDoubleComplex* const       AP[],
                                   int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zspr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_double_complex*)alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)AP,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cspr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 (rocblas_float_complex*)alpha,
                                                                 (rocblas_float_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_float_complex*)AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zspr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 (rocblas_double_complex*)alpha,
                                                                 (rocblas_double_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_double_complex*)AP,
                                                                 strideAP,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_sspr2((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dspr2((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspr2_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspr2_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sspr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  AP,
                                                                  strideAP,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dspr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  AP,
                                                                  strideAP,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssymv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, A, lda, x, incx, beta, y, incy));
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
    return rocBLASStatusToHIPStatus(rocblas_dsymv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, A, lda, x, incx, beta, y, incy));
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
    return rocBLASStatusToHIPStatus(rocblas_csymv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)y,
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
    return rocBLASStatusToHIPStatus(rocblas_zsymv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)y,
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
                                    float*             y[],
                                    int                incy,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssymv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    double*             y[],
                                    int                 incy,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsymv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          beta,
                                                          y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    hipblasComplex*             y[],
                                    int                         incy,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_csymv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    hipblasDoubleComplex*             y[],
                                    int                               incy,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsymv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssymv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsymv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  beta,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csymv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsymv_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(
        rocblas_ssyr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dsyr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda));
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
    return rocBLASStatusToHIPStatus(rocblas_csyr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_float_complex*)alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zsyr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_double_complex*)alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)A,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda, batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_float_complex*)alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)A,
                                                         lda,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_double_complex*)alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)A,
                                                         lda,
                                                         batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 alpha,
                                                                 x,
                                                                 incx,
                                                                 stridex,
                                                                 A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 (rocblas_float_complex*)alpha,
                                                                 (rocblas_float_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_float_complex*)A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyr_strided_batched((rocblas_handle)handle,
                                                                 (rocblas_fill)uplo,
                                                                 n,
                                                                 (rocblas_double_complex*)alpha,
                                                                 (rocblas_double_complex*)x,
                                                                 incx,
                                                                 stridex,
                                                                 (rocblas_double_complex*)A,
                                                                 lda,
                                                                 strideA,
                                                                 batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssyr2(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(rocblas_dsyr2(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(rocblas_csyr2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)A,
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
    return rocBLASStatusToHIPStatus(rocblas_zsyr2((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)A,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          x,
                                                          incx,
                                                          y,
                                                          incy,
                                                          A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          alpha,
                                                          x,
                                                          incx,
                                                          y,
                                                          incy,
                                                          A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyr2_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  alpha,
                                                                  x,
                                                                  incx,
                                                                  stridex,
                                                                  y,
                                                                  incy,
                                                                  stridey,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_float_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyr2_strided_batched((rocblas_handle)handle,
                                                                  (rocblas_fill)uplo,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  (rocblas_double_complex*)y,
                                                                  incy,
                                                                  stridey,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbmv
hipblasStatus_t hipblasStbmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             int                k,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                m,
                             int                k,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                   m,
                             int                   k,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  k,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
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
                             int                         m,
                             int                         k,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztbmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  k,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
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
                                    int                m,
                                    int                k,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          k,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    int                 k,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          k,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    int                         k,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          k,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    int                               k,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztbmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          k,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_stbmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      k,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtbmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      k,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stride_x,
                                      batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctbmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      k,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      stride_a,
                                      (rocblas_float_complex*)x,
                                      incx,
                                      stride_x,
                                      batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztbmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      k,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      stride_a,
                                      (rocblas_double_complex*)x,
                                      incx,
                                      stride_x,
                                      batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_stbsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
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
    return rocBLASStatusToHIPStatus(rocblas_dtbsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
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
    return rocBLASStatusToHIPStatus(rocblas_ctbsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  n,
                                                  k,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
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
    return rocBLASStatusToHIPStatus(rocblas_ztbsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  n,
                                                  k,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_stbsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          n,
                                                          k,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtbsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          n,
                                                          k,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctbsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          n,
                                                          k,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztbsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          n,
                                                          k,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_stbsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      n,
                                      k,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtbsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      n,
                                      k,
                                      A,
                                      lda,
                                      strideA,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctbsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      n,
                                      k,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_float_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztbsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      n,
                                      k,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_double_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpmv
hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_float_complex*)AP,
                                                  (rocblas_float_complex*)x,
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
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztpmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_double_complex*)AP,
                                                  (rocblas_double_complex*)x,
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
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          AP,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          AP,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_float_complex**)AP,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztpmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_double_complex**)AP,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_stpmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      AP,
                                      strideAP,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtpmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      AP,
                                      strideAP,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctpmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_float_complex*)AP,
                                      strideAP,
                                      (rocblas_float_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztpmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_double_complex*)AP,
                                      strideAP,
                                      (rocblas_double_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpsv
hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stpsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtpsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctpsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_float_complex*)AP,
                                                  (rocblas_float_complex*)x,
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
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztpsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_double_complex*)AP,
                                                  (rocblas_double_complex*)x,
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
                                    int                m,
                                    const float* const AP[],
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_stpsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          AP,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtpsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          AP,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctpsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_float_complex**)AP,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztpsv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_double_complex**)AP,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           hipblasStride      strideAP,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_stpsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      AP,
                                      strideAP,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           hipblasStride      strideAP,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtpsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      AP,
                                      strideAP,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           hipblasStride         strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctpsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_float_complex*)AP,
                                      strideAP,
                                      (rocblas_float_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           hipblasStride               strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztpsv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_double_complex*)AP,
                                      strideAP,
                                      (rocblas_double_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmv
hipblasStatus_t hipblasStrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       A,
                             int                lda,
                             float*             x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_strmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtrmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
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
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctrmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
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
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztrmv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
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
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_strmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtrmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctrmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztrmv_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           float*             x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_strmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           hipblasStride      stride_a,
                                           double*            x,
                                           int                incx,
                                           hipblasStride      stridex,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtrmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      A,
                                      lda,
                                      stride_a,
                                      x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           hipblasStride         stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           hipblasStride         stridex,
                                           int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctrmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      stride_a,
                                      (rocblas_float_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           hipblasStride               stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           hipblasStride               stridex,
                                           int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztrmv_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      stride_a,
                                      (rocblas_double_complex*)x,
                                      incx,
                                      stridex,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsv((rocblas_handle)handle,
                                               (rocblas_fill)uplo,
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               A,
                                               lda,
                                               x,
                                               incx)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsv((rocblas_handle)handle,
                                               (rocblas_fill)uplo,
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               A,
                                               lda,
                                               x,
                                               incx)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsv((rocblas_handle)handle,
                                               (rocblas_fill)uplo,
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)x,
                                               incx)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsv((rocblas_handle)handle,
                                               (rocblas_fill)uplo,
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)x,
                                               incx)));
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
                                    int                m,
                                    const float* const A[],
                                    int                lda,
                                    float* const       x[],
                                    int                incx,
                                    int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsv_batched((rocblas_handle)handle,
                                                       (rocblas_fill)uplo,
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       A,
                                                       lda,
                                                       x,
                                                       incx,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const A[],
                                    int                 lda,
                                    double* const       x[],
                                    int                 incx,
                                    int                 batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsv_batched((rocblas_handle)handle,
                                                       (rocblas_fill)uplo,
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       A,
                                                       lda,
                                                       x,
                                                       incx,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const A[],
                                    int                         lda,
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsv_batched((rocblas_handle)handle,
                                                       (rocblas_fill)uplo,
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       (rocblas_float_complex**)A,
                                                       lda,
                                                       (rocblas_float_complex**)x,
                                                       incx,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const A[],
                                    int                               lda,
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsv_batched((rocblas_handle)handle,
                                                       (rocblas_fill)uplo,
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       (rocblas_double_complex**)A,
                                                       lda,
                                                       (rocblas_double_complex**)x,
                                                       incx,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsv_strided_batched((rocblas_handle)handle,
                                                               (rocblas_fill)uplo,
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               x,
                                                               incx,
                                                               stridex,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsv_strided_batched((rocblas_handle)handle,
                                                               (rocblas_fill)uplo,
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               x,
                                                               incx,
                                                               stridex,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsv_strided_batched((rocblas_handle)handle,
                                                               (rocblas_fill)uplo,
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               (rocblas_float_complex*)A,
                                                               lda,
                                                               strideA,
                                                               (rocblas_float_complex*)x,
                                                               incx,
                                                               stridex,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsv_strided_batched((rocblas_handle)handle,
                                                               (rocblas_fill)uplo,
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               (rocblas_double_complex*)A,
                                                               lda,
                                                               strideA,
                                                               (rocblas_double_complex*)x,
                                                               incx,
                                                               stridex,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_cherk((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  beta,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zherk((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  n,
                                                  k,
                                                  alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  beta,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cherk_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          beta,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zherk_batched((rocblas_handle)handle,
                                                          (rocblas_fill)uplo,
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          beta,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_cherk_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      alpha,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (rocblas_float_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zherk_strided_batched((rocblas_handle)handle,
                                      (rocblas_fill)uplo,
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      alpha,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (rocblas_double_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_cherkx((rocblas_handle)handle,
                                                   (rocblas_fill)uplo,
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_float_complex*)alpha,
                                                   (rocblas_float_complex*)A,
                                                   lda,
                                                   (rocblas_float_complex*)B,
                                                   ldb,
                                                   beta,
                                                   (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zherkx((rocblas_handle)handle,
                                                   (rocblas_fill)uplo,
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_double_complex*)alpha,
                                                   (rocblas_double_complex*)A,
                                                   lda,
                                                   (rocblas_double_complex*)B,
                                                   ldb,
                                                   beta,
                                                   (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cherkx_batched((rocblas_handle)handle,
                                                           (rocblas_fill)uplo,
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_float_complex*)alpha,
                                                           (rocblas_float_complex**)A,
                                                           lda,
                                                           (rocblas_float_complex**)B,
                                                           ldb,
                                                           beta,
                                                           (rocblas_float_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zherkx_batched((rocblas_handle)handle,
                                                           (rocblas_fill)uplo,
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_double_complex*)alpha,
                                                           (rocblas_double_complex**)A,
                                                           lda,
                                                           (rocblas_double_complex**)B,
                                                           ldb,
                                                           beta,
                                                           (rocblas_double_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_cherkx_strided_batched((rocblas_handle)handle,
                                       (rocblas_fill)uplo,
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_float_complex*)alpha,
                                       (rocblas_float_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_float_complex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (rocblas_float_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zherkx_strided_batched((rocblas_handle)handle,
                                       (rocblas_fill)uplo,
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_double_complex*)alpha,
                                       (rocblas_double_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_double_complex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (rocblas_double_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_cher2k((rocblas_handle)handle,
                                                   (rocblas_fill)uplo,
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_float_complex*)alpha,
                                                   (rocblas_float_complex*)A,
                                                   lda,
                                                   (rocblas_float_complex*)B,
                                                   ldb,
                                                   beta,
                                                   (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zher2k((rocblas_handle)handle,
                                                   (rocblas_fill)uplo,
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_double_complex*)alpha,
                                                   (rocblas_double_complex*)A,
                                                   lda,
                                                   (rocblas_double_complex*)B,
                                                   ldb,
                                                   beta,
                                                   (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cher2k_batched((rocblas_handle)handle,
                                                           (rocblas_fill)uplo,
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_float_complex*)alpha,
                                                           (rocblas_float_complex**)A,
                                                           lda,
                                                           (rocblas_float_complex**)B,
                                                           ldb,
                                                           beta,
                                                           (rocblas_float_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zher2k_batched((rocblas_handle)handle,
                                                           (rocblas_fill)uplo,
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_double_complex*)alpha,
                                                           (rocblas_double_complex**)A,
                                                           lda,
                                                           (rocblas_double_complex**)B,
                                                           ldb,
                                                           beta,
                                                           (rocblas_double_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_cher2k_strided_batched((rocblas_handle)handle,
                                       (rocblas_fill)uplo,
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_float_complex*)alpha,
                                       (rocblas_float_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_float_complex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (rocblas_float_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zher2k_strided_batched((rocblas_handle)handle,
                                       (rocblas_fill)uplo,
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_double_complex*)alpha,
                                       (rocblas_double_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_double_complex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (rocblas_double_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssymm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
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
    return rocBLASStatusToHIPStatus(rocblas_dsymm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
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
    return rocBLASStatusToHIPStatus(rocblas_csymm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zsymm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssymm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          m,
                                                          n,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsymm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          m,
                                                          n,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csymm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)B,
                                                          ldb,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsymm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)B,
                                                          ldb,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssymm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  m,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  B,
                                                                  ldb,
                                                                  strideB,
                                                                  beta,
                                                                  C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsymm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  m,
                                                                  n,
                                                                  alpha,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  B,
                                                                  ldb,
                                                                  strideB,
                                                                  beta,
                                                                  C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csymm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsymm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssyrk((rocblas_handle)handle,
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_dsyrk((rocblas_handle)handle,
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_csyrk((rocblas_handle)handle,
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  n,
                                                  k,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zsyrk((rocblas_handle)handle,
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  n,
                                                  k,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyrk_batched((rocblas_handle)handle,
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          beta,
                                                          C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyrk_batched((rocblas_handle)handle,
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          beta,
                                                          C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyrk_batched((rocblas_handle)handle,
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyrk_batched((rocblas_handle)handle,
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          n,
                                                          k,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ssyrk_strided_batched((rocblas_handle)handle,
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dsyrk_strided_batched((rocblas_handle)handle,
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_csyrk_strided_batched((rocblas_handle)handle,
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      (rocblas_float_complex*)alpha,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_float_complex*)beta,
                                      (rocblas_float_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zsyrk_strided_batched((rocblas_handle)handle,
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      n,
                                      k,
                                      (rocblas_double_complex*)alpha,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_double_complex*)beta,
                                      (rocblas_double_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssyr2k((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_dsyr2k((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_csyr2k((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_float_complex*)alpha,
                                                   (rocblas_float_complex*)A,
                                                   lda,
                                                   (rocblas_float_complex*)B,
                                                   ldb,
                                                   (rocblas_float_complex*)beta,
                                                   (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zsyr2k((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_double_complex*)alpha,
                                                   (rocblas_double_complex*)A,
                                                   lda,
                                                   (rocblas_double_complex*)B,
                                                   ldb,
                                                   (rocblas_double_complex*)beta,
                                                   (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr2k_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr2k_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyr2k_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_float_complex*)alpha,
                                                           (rocblas_float_complex**)A,
                                                           lda,
                                                           (rocblas_float_complex**)B,
                                                           ldb,
                                                           (rocblas_float_complex*)beta,
                                                           (rocblas_float_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyr2k_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_double_complex*)alpha,
                                                           (rocblas_double_complex**)A,
                                                           lda,
                                                           (rocblas_double_complex**)B,
                                                           ldb,
                                                           (rocblas_double_complex*)beta,
                                                           (rocblas_double_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ssyr2k_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dsyr2k_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_csyr2k_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_float_complex*)alpha,
                                       (rocblas_float_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_float_complex*)B,
                                       ldb,
                                       strideB,
                                       (rocblas_float_complex*)beta,
                                       (rocblas_float_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zsyr2k_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_double_complex*)alpha,
                                       (rocblas_double_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_double_complex*)B,
                                       ldb,
                                       strideB,
                                       (rocblas_double_complex*)beta,
                                       (rocblas_double_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_ssyrkx((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_dsyrkx((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
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
    return rocBLASStatusToHIPStatus(rocblas_csyrkx((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_float_complex*)alpha,
                                                   (rocblas_float_complex*)A,
                                                   lda,
                                                   (rocblas_float_complex*)B,
                                                   ldb,
                                                   (rocblas_float_complex*)beta,
                                                   (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zsyrkx((rocblas_handle)handle,
                                                   hipFillToHCCFill(uplo),
                                                   hipOperationToHCCOperation(transA),
                                                   n,
                                                   k,
                                                   (rocblas_double_complex*)alpha,
                                                   (rocblas_double_complex*)A,
                                                   lda,
                                                   (rocblas_double_complex*)B,
                                                   ldb,
                                                   (rocblas_double_complex*)beta,
                                                   (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ssyrkx_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dsyrkx_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_csyrkx_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_float_complex*)alpha,
                                                           (rocblas_float_complex**)A,
                                                           lda,
                                                           (rocblas_float_complex**)B,
                                                           ldb,
                                                           (rocblas_float_complex*)beta,
                                                           (rocblas_float_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zsyrkx_batched((rocblas_handle)handle,
                                                           hipFillToHCCFill(uplo),
                                                           hipOperationToHCCOperation(transA),
                                                           n,
                                                           k,
                                                           (rocblas_double_complex*)alpha,
                                                           (rocblas_double_complex**)A,
                                                           lda,
                                                           (rocblas_double_complex**)B,
                                                           ldb,
                                                           (rocblas_double_complex*)beta,
                                                           (rocblas_double_complex**)C,
                                                           ldc,
                                                           batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ssyrkx_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dsyrkx_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       alpha,
                                       A,
                                       lda,
                                       strideA,
                                       B,
                                       ldb,
                                       strideB,
                                       beta,
                                       C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_csyrkx_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_float_complex*)alpha,
                                       (rocblas_float_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_float_complex*)B,
                                       ldb,
                                       strideB,
                                       (rocblas_float_complex*)beta,
                                       (rocblas_float_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zsyrkx_strided_batched((rocblas_handle)handle,
                                       hipFillToHCCFill(uplo),
                                       hipOperationToHCCOperation(transA),
                                       n,
                                       k,
                                       (rocblas_double_complex*)alpha,
                                       (rocblas_double_complex*)A,
                                       lda,
                                       strideA,
                                       (rocblas_double_complex*)B,
                                       ldb,
                                       strideB,
                                       (rocblas_double_complex*)beta,
                                       (rocblas_double_complex*)C,
                                       ldc,
                                       strideC,
                                       batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sgeam((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_dgeam((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_cgeam((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zgeam((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sgeam_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          beta,
                                                          B,
                                                          ldb,
                                                          C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_dgeam_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          beta,
                                                          B,
                                                          ldb,
                                                          C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cgeam_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)B,
                                                          ldb,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zgeam_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)B,
                                                          ldb,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_sgeam_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dgeam_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      beta,
                                      B,
                                      ldb,
                                      strideB,
                                      C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_cgeam_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      (rocblas_float_complex*)alpha,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_float_complex*)beta,
                                      (rocblas_float_complex*)B,
                                      ldb,
                                      strideB,
                                      (rocblas_float_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_zgeam_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      (rocblas_double_complex*)alpha,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_double_complex*)beta,
                                      (rocblas_double_complex*)B,
                                      ldb,
                                      strideB,
                                      (rocblas_double_complex*)C,
                                      ldc,
                                      strideC,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_chemm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  n,
                                                  k,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zhemm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  n,
                                                  k,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chemm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          n,
                                                          k,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)B,
                                                          ldb,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhemm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          n,
                                                          k,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)B,
                                                          ldb,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_chemm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  n,
                                                                  k,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  (rocblas_float_complex*)beta,
                                                                  (rocblas_float_complex*)C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zhemm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  hipFillToHCCFill(uplo),
                                                                  n,
                                                                  k,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  (rocblas_double_complex*)beta,
                                                                  (rocblas_double_complex*)C,
                                                                  ldc,
                                                                  strideC,
                                                                  batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmm
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
                             float*             B,
                             int                ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_strmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
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
                             double*            B,
                             int                ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtrmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
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
                             hipblasComplex*       B,
                             int                   ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctrmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)B,
                                                  ldb));
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
                             hipblasDoubleComplex*       B,
                             int                         ldb)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztrmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  hipFillToHCCFill(uplo),
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)B,
                                                  ldb));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmm_batched
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
                                    float* const       B[],
                                    int                ldb,
                                    int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_strmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          B,
                                                          ldb,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    double* const       B[],
                                    int                 ldb,
                                    int                 batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_dtrmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          n,
                                                          alpha,
                                                          A,
                                                          lda,
                                                          B,
                                                          ldb,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    hipblasComplex* const       B[],
                                    int                         ldb,
                                    int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ctrmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)B,
                                                          ldb,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                    hipblasDoubleComplex* const       B[],
                                    int                               ldb,
                                    int                               batchCount)
try
{
    return rocBLASStatusToHIPStatus(rocblas_ztrmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          hipFillToHCCFill(uplo),
                                                          hipOperationToHCCOperation(transA),
                                                          hipDiagonalToHCCDiagonal(diag),
                                                          m,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)B,
                                                          ldb,
                                                          batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// trmm_strided_batched
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
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_strmm_strided_batched((rocblas_handle)handle,
                                      hipSideToHCCSide(side),
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           double*            B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dtrmm_strided_batched((rocblas_handle)handle,
                                      hipSideToHCCSide(side),
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      n,
                                      alpha,
                                      A,
                                      lda,
                                      strideA,
                                      B,
                                      ldb,
                                      strideB,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           hipblasStride         strideB,
                                           int                   batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ctrmm_strided_batched((rocblas_handle)handle,
                                      hipSideToHCCSide(side),
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      n,
                                      (rocblas_float_complex*)alpha,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_float_complex*)B,
                                      ldb,
                                      strideB,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           hipblasStride               strideB,
                                           int                         batchCount)
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_ztrmm_strided_batched((rocblas_handle)handle,
                                      hipSideToHCCSide(side),
                                      hipFillToHCCFill(uplo),
                                      hipOperationToHCCOperation(transA),
                                      hipDiagonalToHCCDiagonal(diag),
                                      m,
                                      n,
                                      (rocblas_double_complex*)alpha,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      strideA,
                                      (rocblas_double_complex*)B,
                                      ldb,
                                      strideB,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsm((rocblas_handle)handle,
                                               hipSideToHCCSide(side),
                                               hipFillToHCCFill(uplo),
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               n,
                                               alpha,
                                               A,
                                               lda,
                                               B,
                                               ldb)));
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
                             double*            A,
                             int                lda,
                             double*            B,
                             int                ldb)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsm((rocblas_handle)handle,
                                               hipSideToHCCSide(side),
                                               hipFillToHCCFill(uplo),
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               n,
                                               alpha,
                                               A,
                                               lda,
                                               B,
                                               ldb)));
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
                             hipblasComplex*       A,
                             int                   lda,
                             hipblasComplex*       B,
                             int                   ldb)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsm((rocblas_handle)handle,
                                               hipSideToHCCSide(side),
                                               hipFillToHCCFill(uplo),
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               n,
                                               (rocblas_float_complex*)alpha,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)B,
                                               ldb)));
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
                             hipblasDoubleComplex*       A,
                             int                         lda,
                             hipblasDoubleComplex*       B,
                             int                         ldb)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsm((rocblas_handle)handle,
                                               hipSideToHCCSide(side),
                                               hipFillToHCCFill(uplo),
                                               hipOperationToHCCOperation(transA),
                                               hipDiagonalToHCCDiagonal(diag),
                                               m,
                                               n,
                                               (rocblas_double_complex*)alpha,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)B,
                                               ldb)));
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
                                    float* const       A[],
                                    int                lda,
                                    float*             B[],
                                    int                ldb,
                                    int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsm_batched((rocblas_handle)handle,
                                                       hipSideToHCCSide(side),
                                                       hipFillToHCCFill(uplo),
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t    handle,
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
                                    int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsm_batched((rocblas_handle)handle,
                                                       hipSideToHCCSide(side),
                                                       hipFillToHCCFill(uplo),
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t       handle,
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
                                    int                   batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsm_batched((rocblas_handle)handle,
                                                       hipSideToHCCSide(side),
                                                       hipFillToHCCFill(uplo),
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       n,
                                                       (rocblas_float_complex*)alpha,
                                                       (rocblas_float_complex**)A,
                                                       lda,
                                                       (rocblas_float_complex**)B,
                                                       ldb,
                                                       batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t             handle,
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
                                    int                         batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsm_batched((rocblas_handle)handle,
                                                       hipSideToHCCSide(side),
                                                       hipFillToHCCFill(uplo),
                                                       hipOperationToHCCOperation(transA),
                                                       hipDiagonalToHCCDiagonal(diag),
                                                       m,
                                                       n,
                                                       (rocblas_double_complex*)alpha,
                                                       (rocblas_double_complex**)A,
                                                       lda,
                                                       (rocblas_double_complex**)B,
                                                       ldb,
                                                       batch_count)));
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
                                           float*             A,
                                           int                lda,
                                           hipblasStride      strideA,
                                           float*             B,
                                           int                ldb,
                                           hipblasStride      strideB,
                                           int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strsm_strided_batched((rocblas_handle)handle,
                                                               hipSideToHCCSide(side),
                                                               hipFillToHCCFill(uplo),
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               n,
                                                               alpha,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               B,
                                                               ldb,
                                                               strideB,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrsmStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrsm_strided_batched((rocblas_handle)handle,
                                                               hipSideToHCCSide(side),
                                                               hipFillToHCCFill(uplo),
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               n,
                                                               alpha,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               B,
                                                               ldb,
                                                               strideB,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrsmStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrsm_strided_batched((rocblas_handle)handle,
                                                               hipSideToHCCSide(side),
                                                               hipFillToHCCFill(uplo),
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               n,
                                                               (rocblas_float_complex*)alpha,
                                                               (rocblas_float_complex*)A,
                                                               lda,
                                                               strideA,
                                                               (rocblas_float_complex*)B,
                                                               ldb,
                                                               strideB,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrsmStridedBatched(hipblasHandle_t             handle,
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
                                           int                         batch_count)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrsm_strided_batched((rocblas_handle)handle,
                                                               hipSideToHCCSide(side),
                                                               hipFillToHCCFill(uplo),
                                                               hipOperationToHCCOperation(transA),
                                                               hipDiagonalToHCCDiagonal(diag),
                                                               m,
                                                               n,
                                                               (rocblas_double_complex*)alpha,
                                                               (rocblas_double_complex*)A,
                                                               lda,
                                                               strideA,
                                                               (rocblas_double_complex*)B,
                                                               ldb,
                                                               strideB,
                                                               batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strtri((rocblas_handle)handle,
                                                hipFillToHCCFill(uplo),
                                                hipDiagonalToHCCDiagonal(diag),
                                                n,
                                                A,
                                                lda,
                                                invA,
                                                ldinvA)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDtrtri(hipblasHandle_t   handle,
                              hipblasFillMode_t uplo,
                              hipblasDiagType_t diag,
                              int               n,
                              const double*     A,
                              int               lda,
                              double*           invA,
                              int               ldinvA)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrtri((rocblas_handle)handle,
                                                hipFillToHCCFill(uplo),
                                                hipDiagonalToHCCDiagonal(diag),
                                                n,
                                                A,
                                                lda,
                                                invA,
                                                ldinvA)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCtrtri(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasDiagType_t     diag,
                              int                   n,
                              const hipblasComplex* A,
                              int                   lda,
                              hipblasComplex*       invA,
                              int                   ldinvA)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrtri((rocblas_handle)handle,
                                                hipFillToHCCFill(uplo),
                                                hipDiagonalToHCCDiagonal(diag),
                                                n,
                                                (rocblas_float_complex*)A,
                                                lda,
                                                (rocblas_float_complex*)invA,
                                                ldinvA)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZtrtri(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasDiagType_t           diag,
                              int                         n,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              hipblasDoubleComplex*       invA,
                              int                         ldinvA)
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrtri((rocblas_handle)handle,
                                                hipFillToHCCFill(uplo),
                                                hipDiagonalToHCCDiagonal(diag),
                                                n,
                                                (rocblas_double_complex*)A,
                                                lda,
                                                (rocblas_double_complex*)invA,
                                                ldinvA)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strtri_batched((rocblas_handle)handle,
                                                        hipFillToHCCFill(uplo),
                                                        hipDiagonalToHCCDiagonal(diag),
                                                        n,
                                                        A,
                                                        lda,
                                                        invA,
                                                        ldinvA,
                                                        batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrtri_batched((rocblas_handle)handle,
                                                        hipFillToHCCFill(uplo),
                                                        hipDiagonalToHCCDiagonal(diag),
                                                        n,
                                                        A,
                                                        lda,
                                                        invA,
                                                        ldinvA,
                                                        batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrtri_batched((rocblas_handle)handle,
                                                        hipFillToHCCFill(uplo),
                                                        hipDiagonalToHCCDiagonal(diag),
                                                        n,
                                                        (rocblas_float_complex**)A,
                                                        lda,
                                                        (rocblas_float_complex**)invA,
                                                        ldinvA,
                                                        batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrtri_batched((rocblas_handle)handle,
                                                        hipFillToHCCFill(uplo),
                                                        hipDiagonalToHCCDiagonal(diag),
                                                        n,
                                                        (rocblas_double_complex**)A,
                                                        lda,
                                                        (rocblas_double_complex**)invA,
                                                        ldinvA,
                                                        batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_strtri_strided_batched((rocblas_handle)handle,
                                                                hipFillToHCCFill(uplo),
                                                                hipDiagonalToHCCDiagonal(diag),
                                                                n,
                                                                A,
                                                                lda,
                                                                stride_A,
                                                                invA,
                                                                ldinvA,
                                                                stride_invA,
                                                                batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_dtrtri_strided_batched((rocblas_handle)handle,
                                                                hipFillToHCCFill(uplo),
                                                                hipDiagonalToHCCDiagonal(diag),
                                                                n,
                                                                A,
                                                                lda,
                                                                stride_A,
                                                                invA,
                                                                ldinvA,
                                                                stride_invA,
                                                                batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ctrtri_strided_batched((rocblas_handle)handle,
                                                                hipFillToHCCFill(uplo),
                                                                hipDiagonalToHCCDiagonal(diag),
                                                                n,
                                                                (rocblas_float_complex*)A,
                                                                lda,
                                                                stride_A,
                                                                (rocblas_float_complex*)invA,
                                                                ldinvA,
                                                                stride_invA,
                                                                batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocblas_ztrtri_strided_batched((rocblas_handle)handle,
                                                                hipFillToHCCFill(uplo),
                                                                hipDiagonalToHCCDiagonal(diag),
                                                                n,
                                                                (rocblas_double_complex*)A,
                                                                lda,
                                                                stride_A,
                                                                (rocblas_double_complex*)invA,
                                                                ldinvA,
                                                                stride_invA,
                                                                batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_sdgmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), m, n, A, lda, x, incx, C, ldc));
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
    return rocBLASStatusToHIPStatus(rocblas_ddgmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), m, n, A, lda, x, incx, C, ldc));
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
    return rocBLASStatusToHIPStatus(rocblas_cdgmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zdgmm((rocblas_handle)handle,
                                                  hipSideToHCCSide(side),
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)C,
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sdgmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          C,
                                                          ldc,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ddgmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          x,
                                                          incx,
                                                          C,
                                                          ldc,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdgmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          m,
                                                          n,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)C,
                                                          ldc,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdgmm_batched((rocblas_handle)handle,
                                                          hipSideToHCCSide(side),
                                                          m,
                                                          n,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)C,
                                                          ldc,
                                                          batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_sdgmm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  m,
                                                                  n,
                                                                  A,
                                                                  lda,
                                                                  stride_A,
                                                                  x,
                                                                  incx,
                                                                  stride_x,
                                                                  C,
                                                                  ldc,
                                                                  stride_C,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_ddgmm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  m,
                                                                  n,
                                                                  A,
                                                                  lda,
                                                                  stride_A,
                                                                  x,
                                                                  incx,
                                                                  stride_x,
                                                                  C,
                                                                  ldc,
                                                                  stride_C,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_cdgmm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  stride_A,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_float_complex*)C,
                                                                  ldc,
                                                                  stride_C,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_zdgmm_strided_batched((rocblas_handle)handle,
                                                                  hipSideToHCCSide(side),
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  stride_A,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stride_x,
                                                                  (rocblas_double_complex*)C,
                                                                  ldc,
                                                                  stride_C,
                                                                  batch_count));
}
catch(...)
{
    return exception_to_hipblas_status();
}

#ifdef __HIP_PLATFORM_SOLVER__
//--------------------------------------------------------------------------------------
//rocSOLVER functions
//--------------------------------------------------------------------------------------

// The following functions are not included in the public API and must be declared

#ifdef __cplusplus
extern "C" {
#endif

rocblas_status rocsolver_sgeqrf_ptr_batched(rocblas_handle    handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            float* const      A[],
                                            const rocblas_int lda,
                                            float* const      ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_dgeqrf_ptr_batched(rocblas_handle    handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            double* const     A[],
                                            const rocblas_int lda,
                                            double* const     ipiv[],
                                            const rocblas_int batch_count);

rocblas_status rocsolver_cgeqrf_ptr_batched(rocblas_handle               handle,
                                            const rocblas_int            m,
                                            const rocblas_int            n,
                                            rocblas_float_complex* const A[],
                                            const rocblas_int            lda,
                                            rocblas_float_complex* const ipiv[],
                                            const rocblas_int            batch_count);

rocblas_status rocsolver_zgeqrf_ptr_batched(rocblas_handle                handle,
                                            const rocblas_int             m,
                                            const rocblas_int             n,
                                            rocblas_double_complex* const A[],
                                            const rocblas_int             lda,
                                            rocblas_double_complex* const ipiv[],
                                            const rocblas_int             batch_count);

#ifdef __cplusplus
}
#endif

// getrf
hipblasStatus_t hipblasSgetrf(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info)
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_sgetrf((rocblas_handle)handle, n, n, A, lda, ipiv, info)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgetrf(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info)
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_dgetrf((rocblas_handle)handle, n, n, A, lda, ipiv, info)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgetrf(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_cgetrf(
        (rocblas_handle)handle, n, n, (rocblas_float_complex*)A, lda, ipiv, info)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgetrf(hipblasHandle_t       handle,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              int*                  ipiv,
                              int*                  info)
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_zgetrf(
        (rocblas_handle)handle, n, n, (rocblas_double_complex*)A, lda, ipiv, info)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetrf_batched(
        (rocblas_handle)handle, n, n, A, lda, ipiv, n, info, batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetrf_batched(
        (rocblas_handle)handle, n, n, A, lda, ipiv, n, info, batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetrf_batched((rocblas_handle)handle,
                                                          n,
                                                          n,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          info,
                                                          batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetrf_batched((rocblas_handle)handle,
                                                          n,
                                                          n,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          info,
                                                          batch_count)));
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
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetrf_strided_batched(
        (rocblas_handle)handle, n, n, A, lda, strideA, ipiv, strideP, info, batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetrf_strided_batched(
        (rocblas_handle)handle, n, n, A, lda, strideA, ipiv, strideP, info, batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetrf_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  n,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  info,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetrf_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  n,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  info,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetrs(
        (rocblas_handle)handle, hipOperationToHCCOperation(trans), n, nrhs, A, lda, ipiv, B, ldb)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetrs(
        (rocblas_handle)handle, hipOperationToHCCOperation(trans), n, nrhs, A, lda, ipiv, B, ldb)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetrs((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  n,
                                                  nrhs,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  ipiv,
                                                  (rocblas_float_complex*)B,
                                                  ldb)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetrs((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(trans),
                                                  n,
                                                  nrhs,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  ipiv,
                                                  (rocblas_double_complex*)B,
                                                  ldb)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else if(batch_count < 0)
        *info = -10;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_sgetrs_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          n,
                                                          nrhs,
                                                          A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          B,
                                                          ldb,
                                                          batch_count)));
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
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else if(batch_count < 0)
        *info = -10;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_dgetrs_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          n,
                                                          nrhs,
                                                          A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          B,
                                                          ldb,
                                                          batch_count)));
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
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else if(batch_count < 0)
        *info = -10;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetrs_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (rocblas_float_complex**)A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          (rocblas_float_complex**)B,
                                                          ldb,
                                                          batch_count)));
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
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -6;
    else if(B == NULL)
        *info = -7;
    else if(ldb < std::max(1, n))
        *info = -8;
    else if(batch_count < 0)
        *info = -10;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetrs_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(trans),
                                                          n,
                                                          nrhs,
                                                          (rocblas_double_complex**)A,
                                                          lda,
                                                          ipiv,
                                                          n,
                                                          (rocblas_double_complex**)B,
                                                          ldb,
                                                          batch_count)));
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -7;
    else if(B == NULL)
        *info = -9;
    else if(ldb < std::max(1, n))
        *info = -10;
    else if(batch_count < 0)
        *info = -13;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_sgetrs_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  n,
                                                                  nrhs,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  B,
                                                                  ldb,
                                                                  strideB,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -7;
    else if(B == NULL)
        *info = -9;
    else if(ldb < std::max(1, n))
        *info = -10;
    else if(batch_count < 0)
        *info = -13;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_dgetrs_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  n,
                                                                  nrhs,
                                                                  A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  B,
                                                                  ldb,
                                                                  strideB,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -7;
    else if(B == NULL)
        *info = -9;
    else if(ldb < std::max(1, n))
        *info = -10;
    else if(batch_count < 0)
        *info = -13;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetrs_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  n,
                                                                  nrhs,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  (rocblas_float_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(n < 0)
        *info = -2;
    else if(nrhs < 0)
        *info = -3;
    else if(A == NULL)
        *info = -4;
    else if(lda < std::max(1, n))
        *info = -5;
    else if(ipiv == NULL)
        *info = -7;
    else if(B == NULL)
        *info = -9;
    else if(ldb < std::max(1, n))
        *info = -10;
    else if(batch_count < 0)
        *info = -13;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetrs_strided_batched((rocblas_handle)handle,
                                                                  hipOperationToHCCOperation(trans),
                                                                  n,
                                                                  nrhs,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  ipiv,
                                                                  strideP,
                                                                  (rocblas_double_complex*)B,
                                                                  ldb,
                                                                  strideB,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetri_outofplace_batched(
        (rocblas_handle)handle, n, A, lda, ipiv, n, C, ldc, info, batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetri_outofplace_batched(
        (rocblas_handle)handle, n, A, lda, ipiv, n, C, ldc, info, batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgetri_outofplace_batched((rocblas_handle)handle,
                                                                     n,
                                                                     (rocblas_float_complex**)A,
                                                                     lda,
                                                                     ipiv,
                                                                     n,
                                                                     (rocblas_float_complex**)C,
                                                                     ldc,
                                                                     info,
                                                                     batch_count)));
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
    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgetri_outofplace_batched((rocblas_handle)handle,
                                                                     n,
                                                                     (rocblas_double_complex**)A,
                                                                     lda,
                                                                     ipiv,
                                                                     n,
                                                                     (rocblas_double_complex**)C,
                                                                     ldc,
                                                                     info,
                                                                     batch_count)));
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
                              float*          tau,
                              int*            info)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_sgeqrf((rocblas_handle)handle, m, n, A, lda, tau)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              double*         A,
                              const int       lda,
                              double*         tau,
                              int*            info)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_dgeqrf((rocblas_handle)handle, m, n, A, lda, tau)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              hipblasComplex* A,
                              const int       lda,
                              hipblasComplex* tau,
                              int*            info)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgeqrf((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)tau)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeqrf(hipblasHandle_t       handle,
                              const int             m,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              hipblasDoubleComplex* tau,
                              int*                  info)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgeqrf((rocblas_handle)handle,
                                                  m,
                                                  n,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)tau)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     float* const    tau[],
                                     int*            info,
                                     const int       batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else if(batch_count < 0)
        *info = -7;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_sgeqrf_ptr_batched((rocblas_handle)handle, m, n, A, lda, tau, batch_count)));
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
                                     double* const   tau[],
                                     int*            info,
                                     const int       batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else if(batch_count < 0)
        *info = -7;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_dgeqrf_ptr_batched((rocblas_handle)handle, m, n, A, lda, tau, batch_count)));
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
                                     hipblasComplex* const tau[],
                                     int*                  info,
                                     const int             batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else if(batch_count < 0)
        *info = -7;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgeqrf_ptr_batched((rocblas_handle)handle,
                                                              m,
                                                              n,
                                                              (rocblas_float_complex**)A,
                                                              lda,
                                                              (rocblas_float_complex**)tau,
                                                              batch_count)));
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
                                     hipblasDoubleComplex* const tau[],
                                     int*                        info,
                                     const int                   batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -5;
    else if(batch_count < 0)
        *info = -7;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgeqrf_ptr_batched((rocblas_handle)handle,
                                                              m,
                                                              n,
                                                              (rocblas_double_complex**)A,
                                                              lda,
                                                              (rocblas_double_complex**)tau,
                                                              batch_count)));
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
                                            float*              tau,
                                            const hipblasStride strideT,
                                            int*                info,
                                            const int           batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -6;
    else if(batch_count < 0)
        *info = -9;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgeqrf_strided_batched(
        (rocblas_handle)handle, m, n, A, lda, strideA, tau, strideT, batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDgeqrfStridedBatched(hipblasHandle_t     handle,
                                            const int           m,
                                            const int           n,
                                            double*             A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            double*             tau,
                                            const hipblasStride strideT,
                                            int*                info,
                                            const int           batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -6;
    else if(batch_count < 0)
        *info = -9;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgeqrf_strided_batched(
        (rocblas_handle)handle, m, n, A, lda, strideA, tau, strideT, batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasCgeqrfStridedBatched(hipblasHandle_t     handle,
                                            const int           m,
                                            const int           n,
                                            hipblasComplex*     A,
                                            const int           lda,
                                            const hipblasStride strideA,
                                            hipblasComplex*     tau,
                                            const hipblasStride strideT,
                                            int*                info,
                                            const int           batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -6;
    else if(batch_count < 0)
        *info = -9;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_cgeqrf_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_float_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_float_complex*)tau,
                                                                  strideT,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasZgeqrfStridedBatched(hipblasHandle_t       handle,
                                            const int             m,
                                            const int             n,
                                            hipblasDoubleComplex* A,
                                            const int             lda,
                                            const hipblasStride   strideA,
                                            hipblasDoubleComplex* tau,
                                            const hipblasStride   strideT,
                                            int*                  info,
                                            const int             batch_count)
try
{
    if(info == NULL)
        return HIPBLAS_STATUS_INVALID_VALUE;
    else if(m < 0)
        *info = -1;
    else if(n < 0)
        *info = -2;
    else if(A == NULL)
        *info = -3;
    else if(lda < std::max(1, m))
        *info = -4;
    else if(tau == NULL)
        *info = -6;
    else if(batch_count < 0)
        *info = -9;
    else
        *info = 0;

    return HIPBLAS_DEMAND_ALLOC(
        rocBLASStatusToHIPStatus(rocsolver_zgeqrf_strided_batched((rocblas_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (rocblas_double_complex*)A,
                                                                  lda,
                                                                  strideA,
                                                                  (rocblas_double_complex*)tau,
                                                                  strideT,
                                                                  batch_count)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_hgemm((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (rocblas_half*)alpha,
                                                  (rocblas_half*)A,
                                                  lda,
                                                  (rocblas_half*)B,
                                                  ldb,
                                                  (rocblas_half*)beta,
                                                  (rocblas_half*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_sgemm((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_dgemm((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_cgemm((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  (rocblas_float_complex*)beta,
                                                  (rocblas_float_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zgemm((rocblas_handle)handle,
                                                  hipOperationToHCCOperation(transa),
                                                  hipOperationToHCCOperation(transb),
                                                  m,
                                                  n,
                                                  k,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  (rocblas_double_complex*)beta,
                                                  (rocblas_double_complex*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_hgemm_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          k,
                                                          (rocblas_half*)alpha,
                                                          (rocblas_half* const*)A,
                                                          lda,
                                                          (rocblas_half* const*)B,
                                                          ldb,
                                                          (rocblas_half*)beta,
                                                          (rocblas_half* const*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_sgemm_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_dgemm_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
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
    return rocBLASStatusToHIPStatus(rocblas_cgemm_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          k,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex* const*)A,
                                                          lda,
                                                          (rocblas_float_complex* const*)B,
                                                          ldb,
                                                          (rocblas_float_complex*)beta,
                                                          (rocblas_float_complex* const*)C,
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
    return rocBLASStatusToHIPStatus(rocblas_zgemm_batched((rocblas_handle)handle,
                                                          hipOperationToHCCOperation(transa),
                                                          hipOperationToHCCOperation(transb),
                                                          m,
                                                          n,
                                                          k,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex* const*)A,
                                                          lda,
                                                          (rocblas_double_complex* const*)B,
                                                          ldb,
                                                          (rocblas_double_complex*)beta,
                                                          (rocblas_double_complex* const*)C,
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
    int bsa_int, bsb_int, bsc_int;
    if(bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
    {
        bsa_int = static_cast<int>(bsa);
        bsb_int = static_cast<int>(bsb);
        bsc_int = static_cast<int>(bsc);
    }
    else
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    return rocBLASStatusToHIPStatus(
        rocblas_hgemm_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      k,
                                      (rocblas_half*)alpha,
                                      (rocblas_half*)A,
                                      lda,
                                      bsa_int,
                                      (rocblas_half*)B,
                                      ldb,
                                      bsb_int,
                                      (rocblas_half*)beta,
                                      (rocblas_half*)C,
                                      ldc,
                                      bsc_int,
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
    int bsa_int, bsb_int, bsc_int;
    if(bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
        try
        {
            bsa_int = static_cast<int>(bsa);
            bsb_int = static_cast<int>(bsb);
            bsc_int = static_cast<int>(bsc);
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }
    else
        try
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }

    return rocBLASStatusToHIPStatus(
        rocblas_sgemm_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa_int,
                                      B,
                                      ldb,
                                      bsb_int,
                                      beta,
                                      C,
                                      ldc,
                                      bsc_int,
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
    int bsa_int, bsb_int, bsc_int;
    if(bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
        try
        {
            bsa_int = static_cast<int>(bsa);
            bsb_int = static_cast<int>(bsb);
            bsc_int = static_cast<int>(bsc);
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }
    else
        try
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }

    return rocBLASStatusToHIPStatus(
        rocblas_dgemm_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa_int,
                                      B,
                                      ldb,
                                      bsb_int,
                                      beta,
                                      C,
                                      ldc,
                                      bsc_int,
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
    int bsa_int, bsb_int, bsc_int;
    if(bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
        try
        {
            bsa_int = static_cast<int>(bsa);
            bsb_int = static_cast<int>(bsb);
            bsc_int = static_cast<int>(bsc);
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }
    else
        try
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }

    return rocBLASStatusToHIPStatus(
        rocblas_cgemm_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      k,
                                      (rocblas_float_complex*)alpha,
                                      (rocblas_float_complex*)A,
                                      lda,
                                      bsa_int,
                                      (rocblas_float_complex*)B,
                                      ldb,
                                      bsb_int,
                                      (rocblas_float_complex*)beta,
                                      (rocblas_float_complex*)C,
                                      ldc,
                                      bsc_int,
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
    int bsa_int, bsb_int, bsc_int;
    if(bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
        try
        {
            bsa_int = static_cast<int>(bsa);
            bsb_int = static_cast<int>(bsb);
            bsc_int = static_cast<int>(bsc);
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }
    else
        try
        {
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        catch(...)
        {
            return exception_to_hipblas_status();
        }

    return rocBLASStatusToHIPStatus(
        rocblas_zgemm_strided_batched((rocblas_handle)handle,
                                      hipOperationToHCCOperation(transa),
                                      hipOperationToHCCOperation(transb),
                                      m,
                                      n,
                                      k,
                                      (rocblas_double_complex*)alpha,
                                      (rocblas_double_complex*)A,
                                      lda,
                                      bsa_int,
                                      (rocblas_double_complex*)B,
                                      ldb,
                                      bsb_int,
                                      (rocblas_double_complex*)beta,
                                      (rocblas_double_complex*)C,
                                      ldc,
                                      bsc_int,
                                      batchCount));
}
catch(...)
{
    return exception_to_hipblas_status();
}

// gemm_ex
// Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
// data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
// format for a given device as documented in rocBLAS.
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
    uint32_t           solution_index = 0;
    rocblas_gemm_flags flags          = rocblas_gemm_flags_none;

    rocblas_status status = rocblas_query_int8_layout_flag((rocblas_handle)handle, &flags);
    if(status != rocblas_status_success)
        return rocBLASStatusToHIPStatus(status);

    return rocBLASStatusToHIPStatus(rocblas_gemm_ex((rocblas_handle)handle,
                                                    hipOperationToHCCOperation(transa),
                                                    hipOperationToHCCOperation(transb),
                                                    m,
                                                    n,
                                                    k,
                                                    alpha,
                                                    A,
                                                    HIPDatatypeToRocblasDatatype(a_type),
                                                    lda,
                                                    B,
                                                    HIPDatatypeToRocblasDatatype(b_type),
                                                    ldb,
                                                    beta,
                                                    C,
                                                    HIPDatatypeToRocblasDatatype(c_type),
                                                    ldc,
                                                    C,
                                                    HIPDatatypeToRocblasDatatype(c_type),
                                                    ldc,
                                                    HIPDatatypeToRocblasDatatype(compute_type),
                                                    HIPGemmAlgoToRocblasGemmAlgo(algo),
                                                    solution_index,
                                                    flags));
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
    uint32_t solution_index = 0;
    uint32_t flags          = 0;

    return rocBLASStatusToHIPStatus(
        rocblas_gemm_batched_ex((rocblas_handle)handle,
                                hipOperationToHCCOperation(transa),
                                hipOperationToHCCOperation(transb),
                                m,
                                n,
                                k,
                                alpha,
                                (void*)A,
                                HIPDatatypeToRocblasDatatype(a_type),
                                lda,
                                (void*)B,
                                HIPDatatypeToRocblasDatatype(b_type),
                                ldb,
                                beta,
                                (void*)C,
                                HIPDatatypeToRocblasDatatype(c_type),
                                ldc,
                                (void*)C,
                                HIPDatatypeToRocblasDatatype(c_type),
                                ldc,
                                batch_count,
                                HIPDatatypeToRocblasDatatype(compute_type),
                                HIPGemmAlgoToRocblasGemmAlgo(algo),
                                solution_index,
                                flags));
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
    uint32_t solution_index = 0;
    uint32_t flags          = 0;

    return rocBLASStatusToHIPStatus(
        rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                        hipOperationToHCCOperation(transa),
                                        hipOperationToHCCOperation(transb),
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        HIPDatatypeToRocblasDatatype(a_type),
                                        lda,
                                        stride_A,
                                        B,
                                        HIPDatatypeToRocblasDatatype(b_type),
                                        ldb,
                                        stride_B,
                                        beta,
                                        C,
                                        HIPDatatypeToRocblasDatatype(c_type),
                                        ldc,
                                        stride_C,
                                        C,
                                        HIPDatatypeToRocblasDatatype(c_type),
                                        ldc,
                                        stride_C,
                                        batch_count,
                                        HIPDatatypeToRocblasDatatype(compute_type),
                                        HIPGemmAlgoToRocblasGemmAlgo(algo),
                                        solution_index,
                                        flags));
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
try
{
    return rocBLASStatusToHIPStatus(rocblas_trsm_ex((rocblas_handle)handle,
                                                    hipSideToHCCSide(side),
                                                    hipFillToHCCFill(uplo),
                                                    hipOperationToHCCOperation(transA),
                                                    hipDiagonalToHCCDiagonal(diag),
                                                    m,
                                                    n,
                                                    alpha,
                                                    A,
                                                    lda,
                                                    B,
                                                    ldb,
                                                    invA,
                                                    invA_size,
                                                    HIPDatatypeToRocblasDatatype(compute_type)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_trsm_batched_ex((rocblas_handle)handle,
                                hipSideToHCCSide(side),
                                hipFillToHCCFill(uplo),
                                hipOperationToHCCOperation(transA),
                                hipDiagonalToHCCDiagonal(diag),
                                m,
                                n,
                                alpha,
                                A,
                                lda,
                                B,
                                ldb,
                                batch_count,
                                invA,
                                invA_size,
                                HIPDatatypeToRocblasDatatype(compute_type)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_trsm_strided_batched_ex((rocblas_handle)handle,
                                        hipSideToHCCSide(side),
                                        hipFillToHCCFill(uplo),
                                        hipOperationToHCCOperation(transA),
                                        hipDiagonalToHCCDiagonal(diag),
                                        m,
                                        n,
                                        alpha,
                                        A,
                                        lda,
                                        stride_A,
                                        B,
                                        ldb,
                                        stride_B,
                                        batch_count,
                                        invA,
                                        invA_size,
                                        stride_invA,
                                        HIPDatatypeToRocblasDatatype(compute_type)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
// {
//     return HIPBLAS_STATUS_NOT_SUPPORTED;
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
//     return HIPBLAS_STATUS_NOT_SUPPORTED;
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
    return rocBLASStatusToHIPStatus(rocblas_axpy_ex((rocblas_handle)handle,
                                                    n,
                                                    alpha,
                                                    HIPDatatypeToRocblasDatatype(alphaType),
                                                    x,
                                                    HIPDatatypeToRocblasDatatype(xType),
                                                    incx,
                                                    y,
                                                    HIPDatatypeToRocblasDatatype(yType),
                                                    incy,
                                                    HIPDatatypeToRocblasDatatype(executionType)));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_axpy_batched_ex((rocblas_handle)handle,
                                n,
                                alpha,
                                HIPDatatypeToRocblasDatatype(alphaType),
                                x,
                                HIPDatatypeToRocblasDatatype(xType),
                                incx,
                                y,
                                HIPDatatypeToRocblasDatatype(yType),
                                incy,
                                batch_count,
                                HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_axpy_strided_batched_ex((rocblas_handle)handle,
                                        n,
                                        alpha,
                                        HIPDatatypeToRocblasDatatype(alphaType),
                                        x,
                                        HIPDatatypeToRocblasDatatype(xType),
                                        incx,
                                        stridex,
                                        y,
                                        HIPDatatypeToRocblasDatatype(yType),
                                        incy,
                                        stridey,
                                        batch_count,
                                        HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_dot_ex((rocblas_handle)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToRocblasDatatype(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToRocblasDatatype(yType),
                                                   incy,
                                                   result,
                                                   HIPDatatypeToRocblasDatatype(resultType),
                                                   HIPDatatypeToRocblasDatatype(executionType)));
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
    return rocBLASStatusToHIPStatus(rocblas_dotc_ex((rocblas_handle)handle,
                                                    n,
                                                    x,
                                                    HIPDatatypeToRocblasDatatype(xType),
                                                    incx,
                                                    y,
                                                    HIPDatatypeToRocblasDatatype(yType),
                                                    incy,
                                                    result,
                                                    HIPDatatypeToRocblasDatatype(resultType),
                                                    HIPDatatypeToRocblasDatatype(executionType)));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dot_batched_ex((rocblas_handle)handle,
                               n,
                               x,
                               HIPDatatypeToRocblasDatatype(xType),
                               incx,
                               y,
                               HIPDatatypeToRocblasDatatype(yType),
                               incy,
                               batch_count,
                               result,
                               HIPDatatypeToRocblasDatatype(resultType),
                               HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dotc_batched_ex((rocblas_handle)handle,
                                n,
                                x,
                                HIPDatatypeToRocblasDatatype(xType),
                                incx,
                                y,
                                HIPDatatypeToRocblasDatatype(yType),
                                incy,
                                batch_count,
                                result,
                                HIPDatatypeToRocblasDatatype(resultType),
                                HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dot_strided_batched_ex((rocblas_handle)handle,
                                       n,
                                       x,
                                       HIPDatatypeToRocblasDatatype(xType),
                                       incx,
                                       stridex,
                                       y,
                                       HIPDatatypeToRocblasDatatype(yType),
                                       incy,
                                       stridey,
                                       batch_count,
                                       result,
                                       HIPDatatypeToRocblasDatatype(resultType),
                                       HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_dotc_strided_batched_ex((rocblas_handle)handle,
                                        n,
                                        x,
                                        HIPDatatypeToRocblasDatatype(xType),
                                        incx,
                                        stridex,
                                        y,
                                        HIPDatatypeToRocblasDatatype(yType),
                                        incy,
                                        stridey,
                                        batch_count,
                                        result,
                                        HIPDatatypeToRocblasDatatype(resultType),
                                        HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_nrm2_ex((rocblas_handle)handle,
                                                    n,
                                                    x,
                                                    HIPDatatypeToRocblasDatatype(xType),
                                                    incx,
                                                    result,
                                                    HIPDatatypeToRocblasDatatype(resultType),
                                                    HIPDatatypeToRocblasDatatype(executionType)));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_nrm2_batched_ex((rocblas_handle)handle,
                                n,
                                x,
                                HIPDatatypeToRocblasDatatype(xType),
                                incx,
                                batch_count,
                                result,
                                HIPDatatypeToRocblasDatatype(resultType),
                                HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_nrm2_strided_batched_ex((rocblas_handle)handle,
                                        n,
                                        x,
                                        HIPDatatypeToRocblasDatatype(xType),
                                        incx,
                                        stridex,
                                        batch_count,
                                        result,
                                        HIPDatatypeToRocblasDatatype(resultType),
                                        HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_rot_ex((rocblas_handle)handle,
                                                   n,
                                                   x,
                                                   HIPDatatypeToRocblasDatatype(xType),
                                                   incx,
                                                   y,
                                                   HIPDatatypeToRocblasDatatype(yType),
                                                   incy,
                                                   c,
                                                   s,
                                                   HIPDatatypeToRocblasDatatype(csType),
                                                   HIPDatatypeToRocblasDatatype(executionType)));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_rot_batched_ex((rocblas_handle)handle,
                               n,
                               x,
                               HIPDatatypeToRocblasDatatype(xType),
                               incx,
                               y,
                               HIPDatatypeToRocblasDatatype(yType),
                               incy,
                               c,
                               s,
                               HIPDatatypeToRocblasDatatype(csType),
                               batch_count,
                               HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_rot_strided_batched_ex((rocblas_handle)handle,
                                       n,
                                       x,
                                       HIPDatatypeToRocblasDatatype(xType),
                                       incx,
                                       stridex,
                                       y,
                                       HIPDatatypeToRocblasDatatype(yType),
                                       incy,
                                       stridey,
                                       c,
                                       s,
                                       HIPDatatypeToRocblasDatatype(csType),
                                       batch_count,
                                       HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
    return rocBLASStatusToHIPStatus(rocblas_scal_ex((rocblas_handle)handle,
                                                    n,
                                                    alpha,
                                                    HIPDatatypeToRocblasDatatype(alphaType),
                                                    x,
                                                    HIPDatatypeToRocblasDatatype(xType),
                                                    incx,
                                                    HIPDatatypeToRocblasDatatype(executionType)));
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_scal_batched_ex((rocblas_handle)handle,
                                n,
                                alpha,
                                HIPDatatypeToRocblasDatatype(alphaType),
                                x,
                                HIPDatatypeToRocblasDatatype(xType),
                                incx,
                                batch_count,
                                HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
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
try
{
    return rocBLASStatusToHIPStatus(
        rocblas_scal_strided_batched_ex((rocblas_handle)handle,
                                        n,
                                        alpha,
                                        HIPDatatypeToRocblasDatatype(alphaType),
                                        x,
                                        HIPDatatypeToRocblasDatatype(xType),
                                        incx,
                                        stridex,
                                        batch_count,
                                        HIPDatatypeToRocblasDatatype(executionType)));
}
catch(...)
{
    return exception_to_hipblas_status();
}

} // extern "C"
