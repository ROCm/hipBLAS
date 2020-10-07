/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "hipblas.h"
#include "limits.h"
#include "rocblas.h"
#ifdef __HIP_PLATFORM_SOLVER__
#include "rocsolver.h"
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
    throw "Non existent OP";
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
    throw "Non existent OP";
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
    throw "Non existent FILL";
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
    throw "Non existent FILL";
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
    throw "Non existent DIAGONAL";
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
    throw "Non existent DIAGONAL";
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
    throw "Non existent SIDE";
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
    throw "Non existent SIDE";
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
    throw "Non existent PointerMode";
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
    throw "Non existent PointerMode";
}

rocblas_datatype HIPDatatypeToRocblasDatatype(hipblasDatatype_t type)
{
    switch(type)
    {
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
    throw "Non existent DataType";
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
    throw "Non existent DataType";
}

rocblas_gemm_algo HIPGemmAlgoToRocblasGemmAlgo(hipblasGemmAlgo_t algo)
{
    switch(algo)
    {
    case HIPBLAS_GEMM_DEFAULT:
        return rocblas_gemm_algo_standard;
    }
    throw "Non existent GemmAlgo";
}

hipblasGemmAlgo_t RocblasGemmAlgoToHIPGemmAlgo(rocblas_gemm_algo algo)
{
    switch(algo)
    {
    case rocblas_gemm_algo_standard:
        return HIPBLAS_GEMM_DEFAULT;
    }
    throw "Non existent GemmAlgo";
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
    throw "Non existent AtomicsMode";
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
    throw "Non existent AtomicsMode";
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
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_invalid_size:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    throw "Unimplemented status";
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
{
    if(!handle)
        return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;

    // Create the rocBLAS handle
    return rocBLASStatusToHIPStatus(rocblas_create_handle((rocblas_handle*)handle));
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
{
    return rocBLASStatusToHIPStatus(rocblas_destroy_handle((rocblas_handle)handle));
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId)
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return rocBLASStatusToHIPStatus(rocblas_set_stream((rocblas_handle)handle, streamId));
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId)
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return rocBLASStatusToHIPStatus(rocblas_get_stream((rocblas_handle)handle, streamId));
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode)
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_pointer_mode((rocblas_handle)handle, HIPPointerModeToRocblasPointerMode(mode)));
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode)
{
    rocblas_pointer_mode rocblas_mode;
    rocblas_status       status = rocblas_get_pointer_mode((rocblas_handle)handle, &rocblas_mode);
    *mode                       = RocblasPointerModeToHIPPointerMode(rocblas_mode);
    return rocBLASStatusToHIPStatus(status);
}

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_set_vector(n, elemSize, x, incx, y, incy));
}

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_get_vector(n, elemSize, x, incx, y, incy));
}

hipblasStatus_t
    hipblasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    return rocBLASStatusToHIPStatus(rocblas_set_matrix(rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t
    hipblasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb)
{
    return rocBLASStatusToHIPStatus(rocblas_get_matrix(rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t hipblasSetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_vector_async(n, elemSize, x, incx, y, incy, stream));
}

hipblasStatus_t hipblasGetVectorAsync(
    int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream)
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_vector_async(n, elemSize, x, incx, y, incy, stream));
}

hipblasStatus_t hipblasSetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
{
    return rocBLASStatusToHIPStatus(
        rocblas_set_matrix_async(rows, cols, elemSize, A, lda, B, ldb, stream));
}

hipblasStatus_t hipblasGetMatrixAsync(
    int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, hipStream_t stream)
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_matrix_async(rows, cols, elemSize, A, lda, B, ldb, stream));
}

// atomics mode
hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode)
{
    return rocBLASStatusToHIPStatus(rocblas_set_atomics_mode(
        (rocblas_handle)handle, HIPAtomicsModeToRocblasAtomicsMode(atomics_mode)));
}

hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode)
{
    return rocBLASStatusToHIPStatus(
        rocblas_get_atomics_mode((rocblas_handle)handle, (rocblas_atomics_mode*)atomics_mode));
}

// amax
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamax((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasIzamax(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamax((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}

// amax_batched
hipblasStatus_t hipblasIsamaxBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_isamax_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasIdamaxBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_idamax_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasIcamaxBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     int*                        result)
{
    return rocBLASStatusToHIPStatus(rocblas_icamax_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}

hipblasStatus_t hipblasIzamaxBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     int*                              result)
{
    return rocBLASStatusToHIPStatus(rocblas_izamax_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}

// amax_strided_batched
hipblasStatus_t hipblasIsamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            int             stridex,
                                            int             batch_count,
                                            int*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamax_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIdamaxStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            int             stridex,
                                            int             batch_count,
                                            int*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamax_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIcamaxStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            int                   stridex,
                                            int                   batch_count,
                                            int*                  result)
{
    return rocBLASStatusToHIPStatus(rocblas_icamax_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIzamaxStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            int                         stridex,
                                            int                         batch_count,
                                            int*                        result)
{
    return rocBLASStatusToHIPStatus(rocblas_izamax_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
}

// amin
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamin((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamin((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamin((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasIzamin(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamin((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}

// amin_batched
hipblasStatus_t hipblasIsaminBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_isamin_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasIdaminBatched(
    hipblasHandle_t handle, int n, const double* const x[], int incx, int batch_count, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_idamin_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasIcaminBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     int*                        result)
{
    return rocBLASStatusToHIPStatus(rocblas_icamin_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}

hipblasStatus_t hipblasIzaminBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     int*                              result)
{
    return rocBLASStatusToHIPStatus(rocblas_izamin_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}

// amin_strided_batched
hipblasStatus_t hipblasIsaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    x,
                                            int             incx,
                                            int             stridex,
                                            int             batch_count,
                                            int*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamin_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIdaminStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const double*   x,
                                            int             incx,
                                            int             stridex,
                                            int             batch_count,
                                            int*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamin_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIcaminStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            int                   stridex,
                                            int                   batch_count,
                                            int*                  result)
{
    return rocBLASStatusToHIPStatus(rocblas_icamin_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasIzaminStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            int                         stridex,
                                            int                         batch_count,
                                            int*                        result)
{
    return rocBLASStatusToHIPStatus(rocblas_izamin_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
}

// asum
hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(rocblas_sasum((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(rocblas_dasum((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_scasum((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dzasum((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}

// asum_batched
hipblasStatus_t hipblasSasumBatched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batch_count, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sasum_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasDasumBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batch_count,
                                    double*             result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dasum_batched((rocblas_handle)handle, n, x, incx, batch_count, result));
}

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batch_count,
                                     float*                      result)
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}

hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batch_count,
                                     double*                           result)
{
    return rocBLASStatusToHIPStatus(rocblas_dzasum_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batch_count, result));
}

// asum_strided_batched
hipblasStatus_t hipblasSasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           int             stridex,
                                           int             batch_count,
                                           float*          result)
{
    return rocBLASStatusToHIPStatus(rocblas_sasum_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasDasumStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           int             stridex,
                                           int             batch_count,
                                           double*         result)
{
    return rocBLASStatusToHIPStatus(rocblas_dasum_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            int                   stridex,
                                            int                   batch_count,
                                            float*                result)
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            int                         stridex,
                                            int                         batch_count,
                                            double*                     result)
{
    return rocBLASStatusToHIPStatus(rocblas_dzasum_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batch_count, result));
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
    return rocBLASStatusToHIPStatus(rocblas_haxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_half*)alpha,
                                                  (rocblas_half*)x,
                                                  incx,
                                                  (rocblas_half*)y,
                                                  incy));
}

hipblasStatus_t hipblasSaxpy(
    hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return rocBLASStatusToHIPStatus(
        rocblas_saxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle,
                             int             n,
                             const double*   alpha,
                             const double*   x,
                             int             incx,
                             double*         y,
                             int             incy)
{
    return rocBLASStatusToHIPStatus(
        rocblas_daxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x,
                             int                   incx,
                             hipblasComplex*       y,
                             int                   incy)
{
    return rocBLASStatusToHIPStatus(rocblas_caxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             hipblasDoubleComplex*       y,
                             int                         incy)
{
    return rocBLASStatusToHIPStatus(rocblas_zaxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
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

hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t    handle,
                                    int                n,
                                    const float*       alpha,
                                    const float* const x[],
                                    int                incx,
                                    float* const       y[],
                                    int                incy,
                                    int                batch_count)
{
    return rocBLASStatusToHIPStatus(
        rocblas_saxpy_batched((rocblas_handle)handle, n, alpha, x, incx, y, incy, batch_count));
}

hipblasStatus_t hipblasDaxpyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double*       alpha,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batch_count)
{
    return rocBLASStatusToHIPStatus(
        rocblas_daxpy_batched((rocblas_handle)handle, n, alpha, x, incx, y, incy, batch_count));
}

hipblasStatus_t hipblasCaxpyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex*       alpha,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batch_count)
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

hipblasStatus_t hipblasZaxpyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex*       alpha,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batch_count)
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

// axpy_strided_batched
hipblasStatus_t hipblasHaxpyStridedBatched(hipblasHandle_t    handle,
                                           int                n,
                                           const hipblasHalf* alpha,
                                           const hipblasHalf* x,
                                           int                incx,
                                           int                stridex,
                                           hipblasHalf*       y,
                                           int                incy,
                                           int                stridey,
                                           int                batch_count)
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

hipblasStatus_t hipblasSaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           const float*    x,
                                           int             incx,
                                           int             stridex,
                                           float*          y,
                                           int             incy,
                                           int             stridey,
                                           int             batch_count)
{
    return rocBLASStatusToHIPStatus(rocblas_saxpy_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count));
}

hipblasStatus_t hipblasDaxpyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           const double*   x,
                                           int             incx,
                                           int             stridex,
                                           double*         y,
                                           int             incy,
                                           int             stridey,
                                           int             batch_count)
{
    return rocBLASStatusToHIPStatus(rocblas_daxpy_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, y, incy, stridey, batch_count));
}

hipblasStatus_t hipblasCaxpyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZaxpyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batch_count)
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

// copy
hipblasStatus_t
    hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_scopy((rocblas_handle)handle, n, x, incx, y, incy));
}

hipblasStatus_t
    hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_dcopy((rocblas_handle)handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasCcopy(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZcopy(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             hipblasDoubleComplex*       y,
                             int                         incy)
{
    return rocBLASStatusToHIPStatus(rocblas_zcopy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
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
    return rocBLASStatusToHIPStatus(
        rocblas_scopy_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    double* const       y[],
                                    int                 incy,
                                    int                 batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dcopy_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    hipblasComplex* const       y[],
                                    int                         incy,
                                    int                         batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    hipblasDoubleComplex* const       y[],
                                    int                               incy,
                                    int                               batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zcopy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}

// copy_strided_batched
hipblasStatus_t hipblasScopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           int             stridex,
                                           float*          y,
                                           int             incy,
                                           int             stridey,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_scopy_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}

hipblasStatus_t hipblasDcopyStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           int             stridex,
                                           double*         y,
                                           int             incy,
                                           int             stridey,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_dcopy_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batchCount)
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

// dot
hipblasStatus_t hipblasHdot(hipblasHandle_t    handle,
                            int                n,
                            const hipblasHalf* x,
                            int                incx,
                            const hipblasHalf* y,
                            int                incy,
                            hipblasHalf*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_hdot((rocblas_handle)handle,
                                                 n,
                                                 (rocblas_half*)x,
                                                 incx,
                                                 (rocblas_half*)y,
                                                 incy,
                                                 (rocblas_half*)result));
}

hipblasStatus_t hipblasBfdot(hipblasHandle_t        handle,
                             int                    n,
                             const hipblasBfloat16* x,
                             int                    incx,
                             const hipblasBfloat16* y,
                             int                    incy,
                             hipblasBfloat16*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_bfdot((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_bfloat16*)x,
                                                  incx,
                                                  (rocblas_bfloat16*)y,
                                                  incy,
                                                  (rocblas_bfloat16*)result));
}

hipblasStatus_t hipblasSdot(hipblasHandle_t handle,
                            int             n,
                            const float*    x,
                            int             incx,
                            const float*    y,
                            int             incy,
                            float*          result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sdot((rocblas_handle)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle,
                            int             n,
                            const double*   x,
                            int             incx,
                            const double*   y,
                            int             incy,
                            double*         result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_ddot((rocblas_handle)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_cdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t       handle,
                             int                   n,
                             const hipblasComplex* x,
                             int                   incx,
                             const hipblasComplex* y,
                             int                   incy,
                             hipblasComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_cdotu((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_zdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)result));
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_zdotu((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)result));
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

hipblasStatus_t hipblasBfdotBatched(hipblasHandle_t              handle,
                                    int                          n,
                                    const hipblasBfloat16* const x[],
                                    int                          incx,
                                    const hipblasBfloat16* const y[],
                                    int                          incy,
                                    int                          batch_count,
                                    hipblasBfloat16*             result)
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

hipblasStatus_t hipblasSdotBatched(hipblasHandle_t    handle,
                                   int                n,
                                   const float* const x[],
                                   int                incx,
                                   const float* const y[],
                                   int                incy,
                                   int                batch_count,
                                   float*             result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sdot_batched((rocblas_handle)handle, n, x, incx, y, incy, batch_count, result));
}

hipblasStatus_t hipblasDdotBatched(hipblasHandle_t     handle,
                                   int                 n,
                                   const double* const x[],
                                   int                 incx,
                                   const double* const y[],
                                   int                 incy,
                                   int                 batch_count,
                                   double*             result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_ddot_batched((rocblas_handle)handle, n, x, incx, y, incy, batch_count, result));
}

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batch_count,
                                    hipblasComplex*             result)
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

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasComplex* const x[],
                                    int                         incx,
                                    const hipblasComplex* const y[],
                                    int                         incy,
                                    int                         batch_count,
                                    hipblasComplex*             result)
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

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batch_count,
                                    hipblasDoubleComplex*             result)
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

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t                   handle,
                                    int                               n,
                                    const hipblasDoubleComplex* const x[],
                                    int                               incx,
                                    const hipblasDoubleComplex* const y[],
                                    int                               incy,
                                    int                               batch_count,
                                    hipblasDoubleComplex*             result)
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

// dot_strided_batched
hipblasStatus_t hipblasHdotStridedBatched(hipblasHandle_t    handle,
                                          int                n,
                                          const hipblasHalf* x,
                                          int                incx,
                                          int                stridex,
                                          const hipblasHalf* y,
                                          int                incy,
                                          int                stridey,
                                          int                batch_count,
                                          hipblasHalf*       result)
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

hipblasStatus_t hipblasBfdotStridedBatched(hipblasHandle_t        handle,
                                           int                    n,
                                           const hipblasBfloat16* x,
                                           int                    incx,
                                           int                    stridex,
                                           const hipblasBfloat16* y,
                                           int                    incy,
                                           int                    stridey,
                                           int                    batch_count,
                                           hipblasBfloat16*       result)
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

hipblasStatus_t hipblasSdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const float*    x,
                                          int             incx,
                                          int             stridex,
                                          const float*    y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count,
                                          float*          result)
{
    return rocBLASStatusToHIPStatus(rocblas_sdot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batch_count, result));
}

hipblasStatus_t hipblasDdotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          const double*   x,
                                          int             incx,
                                          int             stridex,
                                          const double*   y,
                                          int             incy,
                                          int             stridey,
                                          int             batch_count,
                                          double*         result)
{
    return rocBLASStatusToHIPStatus(rocblas_ddot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batch_count, result));
}

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batch_count,
                                           hipblasComplex*       result)
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

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batch_count,
                                           hipblasComplex*       result)
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

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batch_count,
                                           hipblasDoubleComplex*       result)
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

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batch_count,
                                           hipblasDoubleComplex*       result)
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

// nrm2
hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(rocblas_snrm2((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(rocblas_dnrm2((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_scnrm2((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dznrm2((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}

// nrm2_batched
hipblasStatus_t hipblasSnrm2Batched(
    hipblasHandle_t handle, int n, const float* const x[], int incx, int batchCount, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_snrm2_batched((rocblas_handle)handle, n, x, incx, batchCount, result));
}

hipblasStatus_t hipblasDnrm2Batched(hipblasHandle_t     handle,
                                    int                 n,
                                    const double* const x[],
                                    int                 incx,
                                    int                 batchCount,
                                    double*             result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dnrm2_batched((rocblas_handle)handle, n, x, incx, batchCount, result));
}

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t             handle,
                                     int                         n,
                                     const hipblasComplex* const x[],
                                     int                         incx,
                                     int                         batchCount,
                                     float*                      result)
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batchCount, result));
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t                   handle,
                                     int                               n,
                                     const hipblasDoubleComplex* const x[],
                                     int                               incx,
                                     int                               batchCount,
                                     double*                           result)
{
    return rocBLASStatusToHIPStatus(rocblas_dznrm2_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex* const*)x, incx, batchCount, result));
}

// nrm2_strided_batched
hipblasStatus_t hipblasSnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    x,
                                           int             incx,
                                           int             stridex,
                                           int             batchCount,
                                           float*          result)
{
    return rocBLASStatusToHIPStatus(rocblas_snrm2_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batchCount, result));
}

hipblasStatus_t hipblasDnrm2StridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   x,
                                           int             incx,
                                           int             stridex,
                                           int             batchCount,
                                           double*         result)
{
    return rocBLASStatusToHIPStatus(rocblas_dnrm2_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, batchCount, result));
}

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const hipblasComplex* x,
                                            int                   incx,
                                            int                   stridex,
                                            int                   batchCount,
                                            float*                result)
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batchCount, result));
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t             handle,
                                            int                         n,
                                            const hipblasDoubleComplex* x,
                                            int                         incx,
                                            int                         stridex,
                                            int                         batchCount,
                                            double*                     result)
{
    return rocBLASStatusToHIPStatus(rocblas_dznrm2_strided_batched(
        (rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, stridex, batchCount, result));
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
{
    return rocBLASStatusToHIPStatus(
        rocblas_srot((rocblas_handle)handle, n, x, incx, y, incy, c, s));
}

hipblasStatus_t hipblasDrot(hipblasHandle_t handle,
                            int             n,
                            double*         x,
                            int             incx,
                            double*         y,
                            int             incy,
                            const double*   c,
                            const double*   s)
{
    return rocBLASStatusToHIPStatus(
        rocblas_drot((rocblas_handle)handle, n, x, incx, y, incy, c, s));
}

hipblasStatus_t hipblasCrot(hipblasHandle_t       handle,
                            int                   n,
                            hipblasComplex*       x,
                            int                   incx,
                            hipblasComplex*       y,
                            int                   incy,
                            const float*          c,
                            const hipblasComplex* s)
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

hipblasStatus_t hipblasCsrot(hipblasHandle_t handle,
                             int             n,
                             hipblasComplex* x,
                             int             incx,
                             hipblasComplex* y,
                             int             incy,
                             const float*    c,
                             const float*    s)
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

hipblasStatus_t hipblasZrot(hipblasHandle_t             handle,
                            int                         n,
                            hipblasDoubleComplex*       x,
                            int                         incx,
                            hipblasDoubleComplex*       y,
                            int                         incy,
                            const double*               c,
                            const hipblasDoubleComplex* s)
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

hipblasStatus_t hipblasZdrot(hipblasHandle_t       handle,
                             int                   n,
                             hipblasDoubleComplex* x,
                             int                   incx,
                             hipblasDoubleComplex* y,
                             int                   incy,
                             const double*         c,
                             const double*         s)
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
    return rocBLASStatusToHIPStatus(
        rocblas_srot_batched((rocblas_handle)handle, n, x, incx, y, incy, c, s, batchCount));
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
    return rocBLASStatusToHIPStatus(
        rocblas_drot_batched((rocblas_handle)handle, n, x, incx, y, incy, c, s, batchCount));
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

// rot_strided_batched
hipblasStatus_t hipblasSrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          float*          x,
                                          int             incx,
                                          int             stridex,
                                          float*          y,
                                          int             incy,
                                          int             stridey,
                                          const float*    c,
                                          const float*    s,
                                          int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_srot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, c, s, batchCount));
}

hipblasStatus_t hipblasDrotStridedBatched(hipblasHandle_t handle,
                                          int             n,
                                          double*         x,
                                          int             incx,
                                          int             stridex,
                                          double*         y,
                                          int             incy,
                                          int             stridey,
                                          const double*   c,
                                          const double*   s,
                                          int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_drot_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, c, s, batchCount));
}

hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t       handle,
                                          int                   n,
                                          hipblasComplex*       x,
                                          int                   incx,
                                          int                   stridex,
                                          hipblasComplex*       y,
                                          int                   incy,
                                          int                   stridey,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int                   batchCount)
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

hipblasStatus_t hipblasCsrotStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           int             stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           int             stridey,
                                           const float*    c,
                                           const float*    s,
                                           int             batchCount)
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

hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t             handle,
                                          int                         n,
                                          hipblasDoubleComplex*       x,
                                          int                         incx,
                                          int                         stridex,
                                          hipblasDoubleComplex*       y,
                                          int                         incy,
                                          int                         stridey,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int                         batchCount)
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

hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           const double*         c,
                                           const double*         s,
                                           int                   batchCount)
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

// rotg
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    return rocBLASStatusToHIPStatus(rocblas_srotg((rocblas_handle)handle, a, b, c, s));
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s)
{
    return rocBLASStatusToHIPStatus(rocblas_drotg((rocblas_handle)handle, a, b, c, s));
}

hipblasStatus_t hipblasCrotg(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
{
    return rocBLASStatusToHIPStatus(rocblas_crotg((rocblas_handle)handle,
                                                  (rocblas_float_complex*)a,
                                                  (rocblas_float_complex*)b,
                                                  c,
                                                  (rocblas_float_complex*)s));
}

hipblasStatus_t hipblasZrotg(hipblasHandle_t       handle,
                             hipblasDoubleComplex* a,
                             hipblasDoubleComplex* b,
                             double*               c,
                             hipblasDoubleComplex* s)
{
    return rocBLASStatusToHIPStatus(rocblas_zrotg((rocblas_handle)handle,
                                                  (rocblas_double_complex*)a,
                                                  (rocblas_double_complex*)b,
                                                  c,
                                                  (rocblas_double_complex*)s));
}

// rotg_batched
hipblasStatus_t hipblasSrotgBatched(hipblasHandle_t handle,
                                    float* const    a[],
                                    float* const    b[],
                                    float* const    c[],
                                    float* const    s[],
                                    int             batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotg_batched((rocblas_handle)handle, a, b, c, s, batchCount));
}

hipblasStatus_t hipblasDrotgBatched(hipblasHandle_t handle,
                                    double* const   a[],
                                    double* const   b[],
                                    double* const   c[],
                                    double* const   s[],
                                    int             batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotg_batched((rocblas_handle)handle, a, b, c, s, batchCount));
}

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t       handle,
                                    hipblasComplex* const a[],
                                    hipblasComplex* const b[],
                                    float* const          c[],
                                    hipblasComplex* const s[],
                                    int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_crotg_batched((rocblas_handle)handle,
                                                          (rocblas_float_complex**)a,
                                                          (rocblas_float_complex**)b,
                                                          c,
                                                          (rocblas_float_complex**)s,
                                                          batchCount));
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t             handle,
                                    hipblasDoubleComplex* const a[],
                                    hipblasDoubleComplex* const b[],
                                    double* const               c[],
                                    hipblasDoubleComplex* const s[],
                                    int                         batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zrotg_batched((rocblas_handle)handle,
                                                          (rocblas_double_complex**)a,
                                                          (rocblas_double_complex**)b,
                                                          c,
                                                          (rocblas_double_complex**)s,
                                                          batchCount));
}

// rotg_strided_batched
hipblasStatus_t hipblasSrotgStridedBatched(hipblasHandle_t handle,
                                           float*          a,
                                           int             stride_a,
                                           float*          b,
                                           int             stride_b,
                                           float*          c,
                                           int             stride_c,
                                           float*          s,
                                           int             stride_s,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_srotg_strided_batched(
        (rocblas_handle)handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batchCount));
}

hipblasStatus_t hipblasDrotgStridedBatched(hipblasHandle_t handle,
                                           double*         a,
                                           int             stride_a,
                                           double*         b,
                                           int             stride_b,
                                           double*         c,
                                           int             stride_c,
                                           double*         s,
                                           int             stride_s,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_drotg_strided_batched(
        (rocblas_handle)handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batchCount));
}

hipblasStatus_t hipblasCrotgStridedBatched(hipblasHandle_t handle,
                                           hipblasComplex* a,
                                           int             stride_a,
                                           hipblasComplex* b,
                                           int             stride_b,
                                           float*          c,
                                           int             stride_c,
                                           hipblasComplex* s,
                                           int             stride_s,
                                           int             batchCount)
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

hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t       handle,
                                           hipblasDoubleComplex* a,
                                           int                   stride_a,
                                           hipblasDoubleComplex* b,
                                           int                   stride_b,
                                           double*               c,
                                           int                   stride_c,
                                           hipblasDoubleComplex* s,
                                           int                   stride_s,
                                           int                   batchCount)
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

// rotm
hipblasStatus_t hipblasSrotm(
    hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param)
{
    return rocBLASStatusToHIPStatus(
        rocblas_srotm((rocblas_handle)handle, n, x, incx, y, incy, param));
}

hipblasStatus_t hipblasDrotm(
    hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param)
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotm((rocblas_handle)handle, n, x, incx, y, incy, param));
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
    return rocBLASStatusToHIPStatus(
        rocblas_srotm_batched((rocblas_handle)handle, n, x, incx, y, incy, param, batchCount));
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
    return rocBLASStatusToHIPStatus(
        rocblas_drotm_batched((rocblas_handle)handle, n, x, incx, y, incy, param, batchCount));
}

// rotm_strided_batched
hipblasStatus_t hipblasSrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           int             stridex,
                                           float*          y,
                                           int             incy,
                                           int             stridey,
                                           const float*    param,
                                           int             strideparam,
                                           int             batchCount)
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

hipblasStatus_t hipblasDrotmStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           int             stridex,
                                           double*         y,
                                           int             incy,
                                           int             stridey,
                                           const double*   param,
                                           int             strideparam,
                                           int             batchCount)
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

// rotmg
hipblasStatus_t hipblasSrotmg(
    hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    return rocBLASStatusToHIPStatus(rocblas_srotmg((rocblas_handle)handle, d1, d2, x1, y1, param));
}

hipblasStatus_t hipblasDrotmg(
    hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param)
{
    return rocBLASStatusToHIPStatus(rocblas_drotmg((rocblas_handle)handle, d1, d2, x1, y1, param));
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
    return rocBLASStatusToHIPStatus(
        rocblas_srotmg_batched((rocblas_handle)handle, d1, d2, x1, y1, param, batchCount));
}

hipblasStatus_t hipblasDrotmgBatched(hipblasHandle_t     handle,
                                     double* const       d1[],
                                     double* const       d2[],
                                     double* const       x1[],
                                     const double* const y1[],
                                     double* const       param[],
                                     int                 batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_drotmg_batched((rocblas_handle)handle, d1, d2, x1, y1, param, batchCount));
}

// rotmg_strided_batched
hipblasStatus_t hipblasSrotmgStridedBatched(hipblasHandle_t handle,
                                            float*          d1,
                                            int             stride_d1,
                                            float*          d2,
                                            int             stride_d2,
                                            float*          x1,
                                            int             stride_x1,
                                            const float*    y1,
                                            int             stride_y1,
                                            float*          param,
                                            int             strideparam,
                                            int             batchCount)
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

hipblasStatus_t hipblasDrotmgStridedBatched(hipblasHandle_t handle,
                                            double*         d1,
                                            int             stride_d1,
                                            double*         d2,
                                            int             stride_d2,
                                            double*         x1,
                                            int             stride_x1,
                                            const double*   y1,
                                            int             stride_y1,
                                            double*         param,
                                            int             strideparam,
                                            int             batchCount)
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

// scal
hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_sscal((rocblas_handle)handle, n, alpha, x, incx));
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_dscal((rocblas_handle)handle, n, alpha, x, incx));
}

hipblasStatus_t hipblasCscal(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal(
        (rocblas_handle)handle, n, (rocblas_float_complex*)alpha, (rocblas_float_complex*)x, incx));
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(
        rocblas_csscal((rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx));
}

hipblasStatus_t hipblasZscal(hipblasHandle_t             handle,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex*       x,
                             int                         incx)
{
    return rocBLASStatusToHIPStatus(rocblas_zscal((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx));
}

hipblasStatus_t hipblasZdscal(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(
        rocblas_zdscal((rocblas_handle)handle, n, alpha, (rocblas_double_complex*)x, incx));
}

// scal_batched
hipblasStatus_t hipblasSscalBatched(
    hipblasHandle_t handle, int n, const float* alpha, float* const x[], int incx, int batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sscal_batched((rocblas_handle)handle, n, alpha, x, incx, batchCount));
}

hipblasStatus_t hipblasDscalBatched(
    hipblasHandle_t handle, int n, const double* alpha, double* const x[], int incx, int batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dscal_batched((rocblas_handle)handle, n, alpha, x, incx, batchCount));
}

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex* const x[],
                                    int                   incx,
                                    int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex* const*)x,
                                                          incx,
                                                          batchCount));
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t             handle,
                                    int                         n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex* const x[],
                                    int                         incx,
                                    int                         batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex* const*)x,
                                                          incx,
                                                          batchCount));
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t       handle,
                                     int                   n,
                                     const float*          alpha,
                                     hipblasComplex* const x[],
                                     int                   incx,
                                     int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex* const*)x, incx, batchCount));
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t             handle,
                                     int                         n,
                                     const double*               alpha,
                                     hipblasDoubleComplex* const x[],
                                     int                         incx,
                                     int                         batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zdscal_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_double_complex* const*)x, incx, batchCount));
}

// scal_strided_batched
hipblasStatus_t hipblasSscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const float*    alpha,
                                           float*          x,
                                           int             incx,
                                           int             stridex,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_sscal_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, batchCount));
}

hipblasStatus_t hipblasDscalStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           const double*   alpha,
                                           double*         x,
                                           int             incx,
                                           int             stridex,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_dscal_strided_batched(
        (rocblas_handle)handle, n, alpha, x, incx, stridex, batchCount));
}

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  batchCount));
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t             handle,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zscal_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_double_complex*)alpha,
                                                                  (rocblas_double_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  batchCount));
}

hipblasStatus_t hipblasCsscalStridedBatched(hipblasHandle_t handle,
                                            int             n,
                                            const float*    alpha,
                                            hipblasComplex* x,
                                            int             incx,
                                            int             stridex,
                                            int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_strided_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx, stridex, batchCount));
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t       handle,
                                            int                   n,
                                            const double*         alpha,
                                            hipblasDoubleComplex* x,
                                            int                   incx,
                                            int                   stridex,
                                            int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zdscal_strided_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_double_complex*)x, incx, stridex, batchCount));
}

// swap
hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_sswap((rocblas_handle)handle, n, x, incx, y, incy));
}

hipblasStatus_t
    hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_dswap((rocblas_handle)handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasCswap(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_cswap((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZswap(hipblasHandle_t       handle,
                             int                   n,
                             hipblasDoubleComplex* x,
                             int                   incx,
                             hipblasDoubleComplex* y,
                             int                   incy)
{
    return rocBLASStatusToHIPStatus(rocblas_zswap((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
}

// swap_batched
hipblasStatus_t hipblasSswapBatched(
    hipblasHandle_t handle, int n, float* x[], int incx, float* y[], int incy, int batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sswap_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasDswapBatched(
    hipblasHandle_t handle, int n, double* x[], int incx, double* y[], int incy, int batchCount)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dswap_batched((rocblas_handle)handle, n, x, incx, y, incy, batchCount));
}

hipblasStatus_t hipblasCswapBatched(hipblasHandle_t handle,
                                    int             n,
                                    hipblasComplex* x[],
                                    int             incx,
                                    hipblasComplex* y[],
                                    int             incy,
                                    int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_cswap_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t       handle,
                                    int                   n,
                                    hipblasDoubleComplex* x[],
                                    int                   incx,
                                    hipblasDoubleComplex* y[],
                                    int                   incy,
                                    int                   batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zswap_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex**)x,
                                                          incx,
                                                          (rocblas_double_complex**)y,
                                                          incy,
                                                          batchCount));
}

// swap_strided_batched
hipblasStatus_t hipblasSswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           float*          x,
                                           int             incx,
                                           int             stridex,
                                           float*          y,
                                           int             incy,
                                           int             stridey,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_sswap_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}

hipblasStatus_t hipblasDswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           double*         x,
                                           int             incx,
                                           int             stridex,
                                           double*         y,
                                           int             incy,
                                           int             stridey,
                                           int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_dswap_strided_batched(
        (rocblas_handle)handle, n, x, incx, stridex, y, incy, stridey, batchCount));
}

hipblasStatus_t hipblasCswapStridedBatched(hipblasHandle_t handle,
                                           int             n,
                                           hipblasComplex* x,
                                           int             incx,
                                           int             stridex,
                                           hipblasComplex* y,
                                           int             incy,
                                           int             stridey,
                                           int             batchCount)
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

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t       handle,
                                           int                   n,
                                           hipblasDoubleComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           hipblasDoubleComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batchCount)
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
                                           int                stride_a,
                                           const float*       x,
                                           int                incx,
                                           int                stride_x,
                                           const float*       beta,
                                           float*             y,
                                           int                incy,
                                           int                stride_y,
                                           int                batch_count)
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

hipblasStatus_t hipblasDgbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t trans,
                                           int                m,
                                           int                n,
                                           int                kl,
                                           int                ku,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           int                stride_a,
                                           const double*      x,
                                           int                incx,
                                           int                stride_x,
                                           const double*      beta,
                                           double*            y,
                                           int                incy,
                                           int                stride_y,
                                           int                batch_count)
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

hipblasStatus_t hipblasCgbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    trans,
                                           int                   m,
                                           int                   n,
                                           int                   kl,
                                           int                   ku,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           int                   stride_y,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZgbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          trans,
                                           int                         m,
                                           int                         n,
                                           int                         kl,
                                           int                         ku,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stride_y,
                                           int                         batch_count)
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

hipblasStatus_t hipblasCgemvStridedBatched(hipblasHandle_t       handle,
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
                                           int                   batchCount)
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

hipblasStatus_t hipblasZgemvStridedBatched(hipblasHandle_t             handle,
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
                                           int                         incy,
                                           int                         stridey,
                                           int                         batchCount)
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
    return rocBLASStatusToHIPStatus(
        rocblas_sger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
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
    return rocBLASStatusToHIPStatus(rocblas_sger_batched(
        (rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dger_batched(
        (rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda, batchCount));
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

hipblasStatus_t hipblasDgerStridedBatched(hipblasHandle_t handle,
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
                                          int             batchCount)
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

hipblasStatus_t hipblasCgeruStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           int                   strideA,
                                           int                   batchCount)
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

hipblasStatus_t hipblasCgercStridedBatched(hipblasHandle_t       handle,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           int                   strideA,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZgeruStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           int                         strideA,
                                           int                         batchCount)
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

hipblasStatus_t hipblasZgercStridedBatched(hipblasHandle_t             handle,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           int                         strideA,
                                           int                         batchCount)
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

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
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
                                           int                   batchCount)
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

hipblasStatus_t hipblasZhbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batchCount)
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
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   stride_a,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stride_x,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           int                   stride_y,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZhemvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         stride_a,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stride_x,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stride_y,
                                           int                         batch_count)
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

// her
hipblasStatus_t hipblasCher(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda)
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

hipblasStatus_t hipblasZher(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda)
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

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          int                   strideA,
                                          int                   batchCount)
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

hipblasStatus_t hipblasZherStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          int                         strideA,
                                          int                         batchCount)
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

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           int                   strideA,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZher2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           int                         strideA,
                                           int                         batchCount)
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

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* AP,
                                           int                   strideAP,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       y,
                                           int                   incy,
                                           int                   stridey,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZhpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* AP,
                                           int                         strideAP,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       y,
                                           int                         incy,
                                           int                         stridey,
                                           int                         batchCount)
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

// hpr
hipblasStatus_t hipblasChpr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const float*          alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
{
    return rocBLASStatusToHIPStatus(rocblas_chpr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)AP));
}

hipblasStatus_t hipblasZhpr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const double*               alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
{
    return rocBLASStatusToHIPStatus(rocblas_zhpr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)AP));
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
    return rocBLASStatusToHIPStatus(rocblas_chpr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)AP,
                                                         batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_zhpr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)AP,
                                                         batchCount));
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const float*          alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          hipblasComplex*       AP,
                                          int                   strideAP,
                                          int                   batchCount)
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

hipblasStatus_t hipblasZhprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const double*               alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          hipblasDoubleComplex*       AP,
                                          int                         strideAP,
                                          int                         batchCount)
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

hipblasStatus_t hipblasZhpr2(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             int                         n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x,
                             int                         incx,
                             const hipblasDoubleComplex* y,
                             int                         incy,
                             hipblasDoubleComplex*       AP)
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

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           hipblasComplex*       AP,
                                           int                   strideAP,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZhpr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           hipblasDoubleComplex*       AP,
                                           int                         strideAP,
                                           int                         batchCount)
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
{
    return rocBLASStatusToHIPStatus(rocblas_ssbmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
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
{
    return rocBLASStatusToHIPStatus(rocblas_dsbmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, k, alpha, A, lda, x, incx, beta, y, incy));
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

// sbmv_strided_batched
hipblasStatus_t hipblasSsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           int               strideA,
                                           const float*      x,
                                           int               incx,
                                           int               stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

hipblasStatus_t hipblasDsbmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           int               k,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           int               strideA,
                                           const double*     x,
                                           int               incx,
                                           int               stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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
{
    return rocBLASStatusToHIPStatus(rocblas_sspmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, AP, x, incx, beta, y, incy));
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
{
    return rocBLASStatusToHIPStatus(rocblas_dspmv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, AP, x, incx, beta, y, incy));
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

// spmv_strided_batched
hipblasStatus_t hipblasSspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      AP,
                                           int               strideAP,
                                           const float*      x,
                                           int               incx,
                                           int               stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

hipblasStatus_t hipblasDspmvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     AP,
                                           int               strideAP,
                                           const double*     x,
                                           int               incx,
                                           int               stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

// spr
hipblasStatus_t hipblasSspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const float*      alpha,
                            const float*      x,
                            int               incx,
                            float*            AP)
{
    return rocBLASStatusToHIPStatus(
        rocblas_sspr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP));
}

hipblasStatus_t hipblasDspr(hipblasHandle_t   handle,
                            hipblasFillMode_t uplo,
                            int               n,
                            const double*     alpha,
                            const double*     x,
                            int               incx,
                            double*           AP)
{
    return rocBLASStatusToHIPStatus(
        rocblas_dspr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP));
}

hipblasStatus_t hipblasCspr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       AP)
{
    return rocBLASStatusToHIPStatus(rocblas_cspr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_float_complex*)alpha,
                                                 (rocblas_float_complex*)x,
                                                 incx,
                                                 (rocblas_float_complex*)AP));
}

hipblasStatus_t hipblasZspr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       AP)
{
    return rocBLASStatusToHIPStatus(rocblas_zspr((rocblas_handle)handle,
                                                 (rocblas_fill)uplo,
                                                 n,
                                                 (rocblas_double_complex*)alpha,
                                                 (rocblas_double_complex*)x,
                                                 incx,
                                                 (rocblas_double_complex*)AP));
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
    return rocBLASStatusToHIPStatus(rocblas_sspr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP, batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dspr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, AP, batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_cspr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_float_complex*)alpha,
                                                         (rocblas_float_complex**)x,
                                                         incx,
                                                         (rocblas_float_complex**)AP,
                                                         batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_zspr_batched((rocblas_handle)handle,
                                                         (rocblas_fill)uplo,
                                                         n,
                                                         (rocblas_double_complex*)alpha,
                                                         (rocblas_double_complex**)x,
                                                         incx,
                                                         (rocblas_double_complex**)AP,
                                                         batchCount));
}

// spr_strided_batched
hipblasStatus_t hipblasSsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const float*      alpha,
                                          const float*      x,
                                          int               incx,
                                          int               stridex,
                                          float*            AP,
                                          int               strideAP,
                                          int               batchCount)
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

hipblasStatus_t hipblasDsprStridedBatched(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const double*     alpha,
                                          const double*     x,
                                          int               incx,
                                          int               stridex,
                                          double*           AP,
                                          int               strideAP,
                                          int               batchCount)
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

hipblasStatus_t hipblasCsprStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          hipblasComplex*       AP,
                                          int                   strideAP,
                                          int                   batchCount)
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

hipblasStatus_t hipblasZsprStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          hipblasDoubleComplex*       AP,
                                          int                         strideAP,
                                          int                         batchCount)
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
{
    return rocBLASStatusToHIPStatus(
        rocblas_sspr2((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP));
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
{
    return rocBLASStatusToHIPStatus(
        rocblas_dspr2((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP));
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
    return rocBLASStatusToHIPStatus(rocblas_sspr2_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP, batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dspr2_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, AP, batchCount));
}

// spr2_strided_batched
hipblasStatus_t hipblasSspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           int               stridex,
                                           const float*      y,
                                           int               incy,
                                           int               stridey,
                                           float*            AP,
                                           int               strideAP,
                                           int               batchCount)
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

hipblasStatus_t hipblasDspr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           int               stridex,
                                           const double*     y,
                                           int               incy,
                                           int               stridey,
                                           double*           AP,
                                           int               strideAP,
                                           int               batchCount)
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
{
    return rocBLASStatusToHIPStatus(rocblas_ssymv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, A, lda, x, incx, beta, y, incy));
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
{
    return rocBLASStatusToHIPStatus(rocblas_dsymv(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, A, lda, x, incx, beta, y, incy));
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

// symv_strided_batched
hipblasStatus_t hipblasSsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           int               strideA,
                                           const float*      x,
                                           int               incx,
                                           int               stridex,
                                           const float*      beta,
                                           float*            y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

hipblasStatus_t hipblasDsymvStridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           int               strideA,
                                           const double*     x,
                                           int               incx,
                                           int               stridex,
                                           const double*     beta,
                                           double*           y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

hipblasStatus_t hipblasCsymvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
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
                                           int                   batchCount)
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

hipblasStatus_t hipblasZsymvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
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
                                           int                         incy,
                                           int                         stridey,
                                           int                         batchCount)
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
    return rocBLASStatusToHIPStatus(
        rocblas_ssyr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda));
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
    return rocBLASStatusToHIPStatus(
        rocblas_dsyr((rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda));
}

hipblasStatus_t hipblasCsyr(hipblasHandle_t       handle,
                            hipblasFillMode_t     uplo,
                            int                   n,
                            const hipblasComplex* alpha,
                            const hipblasComplex* x,
                            int                   incx,
                            hipblasComplex*       A,
                            int                   lda)
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

hipblasStatus_t hipblasZsyr(hipblasHandle_t             handle,
                            hipblasFillMode_t           uplo,
                            int                         n,
                            const hipblasDoubleComplex* alpha,
                            const hipblasDoubleComplex* x,
                            int                         incx,
                            hipblasDoubleComplex*       A,
                            int                         lda)
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
    return rocBLASStatusToHIPStatus(rocblas_ssyr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda, batchCount));
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
    return rocBLASStatusToHIPStatus(rocblas_dsyr_batched(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, A, lda, batchCount));
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

hipblasStatus_t hipblasCsyrStridedBatched(hipblasHandle_t       handle,
                                          hipblasFillMode_t     uplo,
                                          int                   n,
                                          const hipblasComplex* alpha,
                                          const hipblasComplex* x,
                                          int                   incx,
                                          int                   stridex,
                                          hipblasComplex*       A,
                                          int                   lda,
                                          int                   strideA,
                                          int                   batchCount)
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

hipblasStatus_t hipblasZsyrStridedBatched(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int                         n,
                                          const hipblasDoubleComplex* alpha,
                                          const hipblasDoubleComplex* x,
                                          int                         incx,
                                          int                         stridex,
                                          hipblasDoubleComplex*       A,
                                          int                         lda,
                                          int                         strideA,
                                          int                         batchCount)
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
{
    return rocBLASStatusToHIPStatus(rocblas_ssyr2(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, A, lda));
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
{
    return rocBLASStatusToHIPStatus(rocblas_dsyr2(
        (rocblas_handle)handle, (rocblas_fill)uplo, n, alpha, x, incx, y, incy, A, lda));
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

// syr2_strided_batched
hipblasStatus_t hipblasSsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const float*      alpha,
                                           const float*      x,
                                           int               incx,
                                           int               stridex,
                                           const float*      y,
                                           int               incy,
                                           int               stridey,
                                           float*            A,
                                           int               lda,
                                           int               strideA,
                                           int               batchCount)
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

hipblasStatus_t hipblasDsyr2StridedBatched(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           int               n,
                                           const double*     alpha,
                                           const double*     x,
                                           int               incx,
                                           int               stridex,
                                           const double*     y,
                                           int               incy,
                                           int               stridey,
                                           double*           A,
                                           int               lda,
                                           int               strideA,
                                           int               batchCount)
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

hipblasStatus_t hipblasCsyr2StridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stridex,
                                           const hipblasComplex* y,
                                           int                   incy,
                                           int                   stridey,
                                           hipblasComplex*       A,
                                           int                   lda,
                                           int                   strideA,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZsyr2StridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stridex,
                                           const hipblasDoubleComplex* y,
                                           int                         incy,
                                           int                         stridey,
                                           hipblasDoubleComplex*       A,
                                           int                         lda,
                                           int                         strideA,
                                           int                         batchCount)
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

// tbmv_strided_batched
hipblasStatus_t hipblasStbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           int                stride_a,
                                           float*             x,
                                           int                incx,
                                           int                stride_x,
                                           int                batch_count)
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

hipblasStatus_t hipblasDtbmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           int                stride_a,
                                           double*            x,
                                           int                incx,
                                           int                stride_x,
                                           int                batch_count)
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

hipblasStatus_t hipblasCtbmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stride_x,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZtbmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stride_x,
                                           int                         batch_count)
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

// tbsv_strided_batched
hipblasStatus_t hipblasStbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const float*       A,
                                           int                lda,
                                           int                strideA,
                                           float*             x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasDtbsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                n,
                                           int                k,
                                           const double*      A,
                                           int                lda,
                                           int                strideA,
                                           double*            x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasCtbsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZtbsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batchCount)
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

// tpmv
hipblasStatus_t hipblasStpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
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

hipblasStatus_t hipblasDtpmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
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

hipblasStatus_t hipblasCtpmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
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

hipblasStatus_t hipblasZtpmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
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

hipblasStatus_t hipblasDtpmvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
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

hipblasStatus_t hipblasCtpmvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
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

hipblasStatus_t hipblasZtpmvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
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

// tpmv_strided_batched
hipblasStatus_t hipblasStpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           int                strideAP,
                                           float*             x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasDtpmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           int                strideAP,
                                           double*            x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasCtpmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           int                   strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZtpmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           int                         strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batchCount)
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

// tpsv
hipblasStatus_t hipblasStpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const float*       AP,
                             float*             x,
                             int                incx)
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

hipblasStatus_t hipblasDtpsv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      AP,
                             double*            x,
                             int                incx)
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

hipblasStatus_t hipblasCtpsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* AP,
                             hipblasComplex*       x,
                             int                   incx)
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

hipblasStatus_t hipblasZtpsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* AP,
                             hipblasDoubleComplex*       x,
                             int                         incx)
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

hipblasStatus_t hipblasDtpsvBatched(hipblasHandle_t     handle,
                                    hipblasFillMode_t   uplo,
                                    hipblasOperation_t  transA,
                                    hipblasDiagType_t   diag,
                                    int                 m,
                                    const double* const AP[],
                                    double* const       x[],
                                    int                 incx,
                                    int                 batchCount)
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

hipblasStatus_t hipblasCtpsvBatched(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int                         m,
                                    const hipblasComplex* const AP[],
                                    hipblasComplex* const       x[],
                                    int                         incx,
                                    int                         batchCount)
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

hipblasStatus_t hipblasZtpsvBatched(hipblasHandle_t                   handle,
                                    hipblasFillMode_t                 uplo,
                                    hipblasOperation_t                transA,
                                    hipblasDiagType_t                 diag,
                                    int                               m,
                                    const hipblasDoubleComplex* const AP[],
                                    hipblasDoubleComplex* const       x[],
                                    int                               incx,
                                    int                               batchCount)
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

// tpsv_strided_batched
hipblasStatus_t hipblasStpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       AP,
                                           int                strideAP,
                                           float*             x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasDtpsvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      AP,
                                           int                strideAP,
                                           double*            x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasCtpsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* AP,
                                           int                   strideAP,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZtpsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* AP,
                                           int                         strideAP,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batchCount)
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

hipblasStatus_t hipblasDtrmv(hipblasHandle_t    handle,
                             hipblasFillMode_t  uplo,
                             hipblasOperation_t transA,
                             hipblasDiagType_t  diag,
                             int                m,
                             const double*      A,
                             int                lda,
                             double*            x,
                             int                incx)
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

hipblasStatus_t hipblasCtrmv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
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

hipblasStatus_t hipblasZtrmv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
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

// trmv_strided_batched
hipblasStatus_t hipblasStrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const float*       A,
                                           int                lda,
                                           int                stride_a,
                                           float*             x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasDtrmvStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           hipblasDiagType_t  diag,
                                           int                m,
                                           const double*      A,
                                           int                lda,
                                           int                stride_a,
                                           double*            x,
                                           int                incx,
                                           int                stridex,
                                           int                batchCount)
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

hipblasStatus_t hipblasCtrmvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   stride_a,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZtrmvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         stride_a,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batchCount)
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

hipblasStatus_t hipblasCtrsv(hipblasHandle_t       handle,
                             hipblasFillMode_t     uplo,
                             hipblasOperation_t    transA,
                             hipblasDiagType_t     diag,
                             int                   m,
                             const hipblasComplex* A,
                             int                   lda,
                             hipblasComplex*       x,
                             int                   incx)
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

hipblasStatus_t hipblasZtrsv(hipblasHandle_t             handle,
                             hipblasFillMode_t           uplo,
                             hipblasOperation_t          transA,
                             hipblasDiagType_t           diag,
                             int                         m,
                             const hipblasDoubleComplex* A,
                             int                         lda,
                             hipblasDoubleComplex*       x,
                             int                         incx)
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

// trsv_strided_batched
hipblasStatus_t hipblasStrsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
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

hipblasStatus_t hipblasDtrsvStridedBatched(hipblasHandle_t    handle,
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
                                           int                batch_count)
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

hipblasStatus_t hipblasCtrsvStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           hipblasDiagType_t     diag,
                                           int                   m,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           hipblasComplex*       x,
                                           int                   incx,
                                           int                   stridex,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZtrsvStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int                         m,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           hipblasDoubleComplex*       x,
                                           int                         incx,
                                           int                         stridex,
                                           int                         batch_count)
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

// herk_strided_batched
hipblasStatus_t hipblasCherkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const float*          alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const float*          beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZherkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const double*               alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const double*               beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
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

// herkx_strided_batched
hipblasStatus_t hipblasCherkxStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            int                   strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            int                   strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            int                   strideC,
                                            int                   batchCount)
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

hipblasStatus_t hipblasZherkxStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            int                         strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            int                         strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            int                         strideC,
                                            int                         batchCount)
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

// her2k_strided_batched
hipblasStatus_t hipblasCher2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            int                   strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            int                   strideB,
                                            const float*          beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            int                   strideC,
                                            int                   batchCount)
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

hipblasStatus_t hipblasZher2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            int                         strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            int                         strideB,
                                            const double*               beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            int                         strideC,
                                            int                         batchCount)
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

// symm_strided_batched
hipblasStatus_t hipblasSsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const float*      alpha,
                                           const float*      A,
                                           int               lda,
                                           int               strideA,
                                           const float*      B,
                                           int               ldb,
                                           int               strideB,
                                           const float*      beta,
                                           float*            C,
                                           int               ldc,
                                           int               strideC,
                                           int               batchCount)
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

hipblasStatus_t hipblasDsymmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           hipblasFillMode_t uplo,
                                           int               m,
                                           int               n,
                                           const double*     alpha,
                                           const double*     A,
                                           int               lda,
                                           int               strideA,
                                           const double*     B,
                                           int               ldb,
                                           int               strideB,
                                           const double*     beta,
                                           double*           C,
                                           int               ldc,
                                           int               strideC,
                                           int               batchCount)
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

hipblasStatus_t hipblasCsymmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZsymmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
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

// syrk_strided_batched
hipblasStatus_t hipblasSsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           int                strideA,
                                           const float*       beta,
                                           float*             C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
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

hipblasStatus_t hipblasDsyrkStridedBatched(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           int                strideA,
                                           const double*      beta,
                                           double*            C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
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

hipblasStatus_t hipblasCsyrkStridedBatched(hipblasHandle_t       handle,
                                           hipblasFillMode_t     uplo,
                                           hipblasOperation_t    transA,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZsyrkStridedBatched(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
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

// syr2k_strided_batched
hipblasStatus_t hipblasSsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            int                strideA,
                                            const float*       B,
                                            int                ldb,
                                            int                strideB,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc,
                                            int                strideC,
                                            int                batchCount)
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

hipblasStatus_t hipblasDsyr2kStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            int                strideA,
                                            const double*      B,
                                            int                ldb,
                                            int                strideB,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc,
                                            int                strideC,
                                            int                batchCount)
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

hipblasStatus_t hipblasCsyr2kStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            int                   strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            int                   strideB,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            int                   strideC,
                                            int                   batchCount)
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

hipblasStatus_t hipblasZsyr2kStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            int                         strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            int                         strideB,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            int                         strideC,
                                            int                         batchCount)
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

// syrkx_strided_batched
hipblasStatus_t hipblasSsyrkxStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const float*       alpha,
                                            const float*       A,
                                            int                lda,
                                            int                strideA,
                                            const float*       B,
                                            int                ldb,
                                            int                strideB,
                                            const float*       beta,
                                            float*             C,
                                            int                ldc,
                                            int                strideC,
                                            int                batchCount)
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

hipblasStatus_t hipblasDsyrkxStridedBatched(hipblasHandle_t    handle,
                                            hipblasFillMode_t  uplo,
                                            hipblasOperation_t transA,
                                            int                n,
                                            int                k,
                                            const double*      alpha,
                                            const double*      A,
                                            int                lda,
                                            int                strideA,
                                            const double*      B,
                                            int                ldb,
                                            int                strideB,
                                            const double*      beta,
                                            double*            C,
                                            int                ldc,
                                            int                strideC,
                                            int                batchCount)
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

hipblasStatus_t hipblasCsyrkxStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasOperation_t    transA,
                                            int                   n,
                                            int                   k,
                                            const hipblasComplex* alpha,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            int                   strideA,
                                            const hipblasComplex* B,
                                            int                   ldb,
                                            int                   strideB,
                                            const hipblasComplex* beta,
                                            hipblasComplex*       C,
                                            int                   ldc,
                                            int                   strideC,
                                            int                   batchCount)
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

hipblasStatus_t hipblasZsyrkxStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int                         n,
                                            int                         k,
                                            const hipblasDoubleComplex* alpha,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            int                         strideA,
                                            const hipblasDoubleComplex* B,
                                            int                         ldb,
                                            int                         strideB,
                                            const hipblasDoubleComplex* beta,
                                            hipblasDoubleComplex*       C,
                                            int                         ldc,
                                            int                         strideC,
                                            int                         batchCount)
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

// geam_strided_batched
hipblasStatus_t hipblasSgeamStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           const float*       alpha,
                                           const float*       A,
                                           int                lda,
                                           int                strideA,
                                           const float*       beta,
                                           const float*       B,
                                           int                ldb,
                                           int                strideB,
                                           float*             C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
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

hipblasStatus_t hipblasDgeamStridedBatched(hipblasHandle_t    handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int                m,
                                           int                n,
                                           const double*      alpha,
                                           const double*      A,
                                           int                lda,
                                           int                strideA,
                                           const double*      beta,
                                           const double*      B,
                                           int                ldb,
                                           int                strideB,
                                           double*            C,
                                           int                ldc,
                                           int                strideC,
                                           int                batchCount)
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

hipblasStatus_t hipblasCgeamStridedBatched(hipblasHandle_t       handle,
                                           hipblasOperation_t    transa,
                                           hipblasOperation_t    transb,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* beta,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZgeamStridedBatched(hipblasHandle_t             handle,
                                           hipblasOperation_t          transa,
                                           hipblasOperation_t          transb,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* beta,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
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

// hemm_strided_batched
hipblasStatus_t hipblasChemmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           hipblasFillMode_t     uplo,
                                           int                   n,
                                           int                   k,
                                           const hipblasComplex* alpha,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   strideA,
                                           const hipblasComplex* B,
                                           int                   ldb,
                                           int                   strideB,
                                           const hipblasComplex* beta,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   strideC,
                                           int                   batchCount)
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

hipblasStatus_t hipblasZhemmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int                         n,
                                           int                         k,
                                           const hipblasDoubleComplex* alpha,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         strideA,
                                           const hipblasDoubleComplex* B,
                                           int                         ldb,
                                           int                         strideB,
                                           const hipblasDoubleComplex* beta,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         strideC,
                                           int                         batchCount)
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
                                           int                strideA,
                                           float*             B,
                                           int                ldb,
                                           int                strideB,
                                           int                batchCount)
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
                                           int                strideA,
                                           double*            B,
                                           int                ldb,
                                           int                strideB,
                                           int                batchCount)
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
                                           int                   strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           int                   strideB,
                                           int                   batchCount)
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
                                           int                         strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           int                         strideB,
                                           int                         batchCount)
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
                                           int                strideA,
                                           float*             B,
                                           int                ldb,
                                           int                strideB,
                                           int                batch_count)
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
                                           int                strideA,
                                           double*            B,
                                           int                ldb,
                                           int                strideB,
                                           int                batch_count)
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
                                           int                   strideA,
                                           hipblasComplex*       B,
                                           int                   ldb,
                                           int                   strideB,
                                           int                   batch_count)
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
                                           int                         strideA,
                                           hipblasDoubleComplex*       B,
                                           int                         ldb,
                                           int                         strideB,
                                           int                         batch_count)
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

hipblasStatus_t hipblasDtrtri(hipblasHandle_t   handle,
                              hipblasFillMode_t uplo,
                              hipblasDiagType_t diag,
                              int               n,
                              const double*     A,
                              int               lda,
                              double*           invA,
                              int               ldinvA)
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

hipblasStatus_t hipblasCtrtri(hipblasHandle_t       handle,
                              hipblasFillMode_t     uplo,
                              hipblasDiagType_t     diag,
                              int                   n,
                              const hipblasComplex* A,
                              int                   lda,
                              hipblasComplex*       invA,
                              int                   ldinvA)
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

hipblasStatus_t hipblasZtrtri(hipblasHandle_t             handle,
                              hipblasFillMode_t           uplo,
                              hipblasDiagType_t           diag,
                              int                         n,
                              const hipblasDoubleComplex* A,
                              int                         lda,
                              hipblasDoubleComplex*       invA,
                              int                         ldinvA)
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

// trtri_strided_batched
hipblasStatus_t hipblasStrtriStridedBatched(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            hipblasDiagType_t diag,
                                            int               n,
                                            const float*      A,
                                            int               lda,
                                            int               stride_A,
                                            float*            invA,
                                            int               ldinvA,
                                            int               stride_invA,
                                            int               batch_count)
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

hipblasStatus_t hipblasDtrtriStridedBatched(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            hipblasDiagType_t diag,
                                            int               n,
                                            const double*     A,
                                            int               lda,
                                            int               stride_A,
                                            double*           invA,
                                            int               ldinvA,
                                            int               stride_invA,
                                            int               batch_count)
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

hipblasStatus_t hipblasCtrtriStridedBatched(hipblasHandle_t       handle,
                                            hipblasFillMode_t     uplo,
                                            hipblasDiagType_t     diag,
                                            int                   n,
                                            const hipblasComplex* A,
                                            int                   lda,
                                            int                   stride_A,
                                            hipblasComplex*       invA,
                                            int                   ldinvA,
                                            int                   stride_invA,
                                            int                   batch_count)
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

hipblasStatus_t hipblasZtrtriStridedBatched(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasDiagType_t           diag,
                                            int                         n,
                                            const hipblasDoubleComplex* A,
                                            int                         lda,
                                            int                         stride_A,
                                            hipblasDoubleComplex*       invA,
                                            int                         ldinvA,
                                            int                         stride_invA,
                                            int                         batch_count)
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
{
    return rocBLASStatusToHIPStatus(rocblas_sdgmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), m, n, A, lda, x, incx, C, ldc));
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
{
    return rocBLASStatusToHIPStatus(rocblas_ddgmm(
        (rocblas_handle)handle, hipSideToHCCSide(side), m, n, A, lda, x, incx, C, ldc));
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

// dgmm_strided_batched
hipblasStatus_t hipblasSdgmmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           int               m,
                                           int               n,
                                           const float*      A,
                                           int               lda,
                                           int               stride_A,
                                           const float*      x,
                                           int               incx,
                                           int               stride_x,
                                           float*            C,
                                           int               ldc,
                                           int               stride_C,
                                           int               batch_count)
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

hipblasStatus_t hipblasDdgmmStridedBatched(hipblasHandle_t   handle,
                                           hipblasSideMode_t side,
                                           int               m,
                                           int               n,
                                           const double*     A,
                                           int               lda,
                                           int               stride_A,
                                           const double*     x,
                                           int               incx,
                                           int               stride_x,
                                           double*           C,
                                           int               ldc,
                                           int               stride_C,
                                           int               batch_count)
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

hipblasStatus_t hipblasCdgmmStridedBatched(hipblasHandle_t       handle,
                                           hipblasSideMode_t     side,
                                           int                   m,
                                           int                   n,
                                           const hipblasComplex* A,
                                           int                   lda,
                                           int                   stride_A,
                                           const hipblasComplex* x,
                                           int                   incx,
                                           int                   stride_x,
                                           hipblasComplex*       C,
                                           int                   ldc,
                                           int                   stride_C,
                                           int                   batch_count)
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

hipblasStatus_t hipblasZdgmmStridedBatched(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           int                         m,
                                           int                         n,
                                           const hipblasDoubleComplex* A,
                                           int                         lda,
                                           int                         stride_A,
                                           const hipblasDoubleComplex* x,
                                           int                         incx,
                                           int                         stride_x,
                                           hipblasDoubleComplex*       C,
                                           int                         ldc,
                                           int                         stride_C,
                                           int                         batch_count)
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

rocblas_status rocsolver_sgetri_outofplace_batched(rocblas_handle       handle,
                                                   const rocblas_int    n,
                                                   float* const         A[],
                                                   const rocblas_int    lda,
                                                   rocblas_int*         ipiv,
                                                   const rocblas_stride strideP,
                                                   float* const         C[],
                                                   const rocblas_int    ldc,
                                                   rocblas_int*         info,
                                                   const rocblas_int    batch_count);

rocblas_status rocsolver_dgetri_outofplace_batched(rocblas_handle       handle,
                                                   const rocblas_int    n,
                                                   double* const        A[],
                                                   const rocblas_int    lda,
                                                   rocblas_int*         ipiv,
                                                   const rocblas_stride strideP,
                                                   double* const        C[],
                                                   const rocblas_int    ldc,
                                                   rocblas_int*         info,
                                                   const rocblas_int    batch_count);

rocblas_status rocsolver_cgetri_outofplace_batched(rocblas_handle               handle,
                                                   const rocblas_int            n,
                                                   rocblas_float_complex* const A[],
                                                   const rocblas_int            lda,
                                                   rocblas_int*                 ipiv,
                                                   const rocblas_stride         strideP,
                                                   rocblas_float_complex* const C[],
                                                   const rocblas_int            ldc,
                                                   rocblas_int*                 info,
                                                   const rocblas_int            batch_count);

rocblas_status rocsolver_zgetri_outofplace_batched(rocblas_handle                handle,
                                                   const rocblas_int             n,
                                                   rocblas_double_complex* const A[],
                                                   const rocblas_int             lda,
                                                   rocblas_int*                  ipiv,
                                                   const rocblas_stride          strideP,
                                                   rocblas_double_complex* const C[],
                                                   const rocblas_int             ldc,
                                                   rocblas_int*                  info,
                                                   const rocblas_int             batch_count);

#ifdef __cplusplus
}
#endif

// getrf
hipblasStatus_t hipblasSgetrf(
    hipblasHandle_t handle, const int n, float* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_sgetrf((rocblas_handle)handle, n, n, A, lda, ipiv, info)));
}

hipblasStatus_t hipblasDgetrf(
    hipblasHandle_t handle, const int n, double* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(
        rocsolver_dgetrf((rocblas_handle)handle, n, n, A, lda, ipiv, info)));
}

hipblasStatus_t hipblasCgetrf(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_cgetrf(
        (rocblas_handle)handle, n, n, (rocblas_float_complex*)A, lda, ipiv, info)));
}

hipblasStatus_t hipblasZgetrf(hipblasHandle_t       handle,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              int*                  ipiv,
                              int*                  info)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_zgetrf(
        (rocblas_handle)handle, n, n, (rocblas_double_complex*)A, lda, ipiv, info)));
}

// getrf_batched
hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     int*            ipiv,
                                     int*            info,
                                     const int       batch_count)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetrf_batched(
        (rocblas_handle)handle, n, n, A, lda, ipiv, n, info, batch_count)));
}

hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     int*            ipiv,
                                     int*            info,
                                     const int       batch_count)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetrf_batched(
        (rocblas_handle)handle, n, n, A, lda, ipiv, n, info, batch_count)));
}

hipblasStatus_t hipblasCgetrfBatched(hipblasHandle_t       handle,
                                     const int             n,
                                     hipblasComplex* const A[],
                                     const int             lda,
                                     int*                  ipiv,
                                     int*                  info,
                                     const int             batch_count)
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

hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t             handle,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     int*                        ipiv,
                                     int*                        info,
                                     const int                   batch_count)
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

// getrf_strided_batched
hipblasStatus_t hipblasSgetrfStridedBatched(hipblasHandle_t handle,
                                            const int       n,
                                            float*          A,
                                            const int       lda,
                                            const int       strideA,
                                            int*            ipiv,
                                            const int       strideP,
                                            int*            info,
                                            const int       batch_count)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetrf_strided_batched(
        (rocblas_handle)handle, n, n, A, lda, strideA, ipiv, strideP, info, batch_count)));
}

hipblasStatus_t hipblasDgetrfStridedBatched(hipblasHandle_t handle,
                                            const int       n,
                                            double*         A,
                                            const int       lda,
                                            const int       strideA,
                                            int*            ipiv,
                                            const int       strideP,
                                            int*            info,
                                            const int       batch_count)
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetrf_strided_batched(
        (rocblas_handle)handle, n, n, A, lda, strideA, ipiv, strideP, info, batch_count)));
}

hipblasStatus_t hipblasCgetrfStridedBatched(hipblasHandle_t handle,
                                            const int       n,
                                            hipblasComplex* A,
                                            const int       lda,
                                            const int       strideA,
                                            int*            ipiv,
                                            const int       strideP,
                                            int*            info,
                                            const int       batch_count)
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

hipblasStatus_t hipblasZgetrfStridedBatched(hipblasHandle_t       handle,
                                            const int             n,
                                            hipblasDoubleComplex* A,
                                            const int             lda,
                                            const int             strideA,
                                            int*                  ipiv,
                                            const int             strideP,
                                            int*                  info,
                                            const int             batch_count)
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

// getrs_strided_batched
hipblasStatus_t hipblasSgetrsStridedBatched(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            float*                   A,
                                            const int                lda,
                                            const int                strideA,
                                            const int*               ipiv,
                                            const int                strideP,
                                            float*                   B,
                                            const int                ldb,
                                            const int                strideB,
                                            int*                     info,
                                            const int                batch_count)
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

hipblasStatus_t hipblasDgetrsStridedBatched(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            double*                  A,
                                            const int                lda,
                                            const int                strideA,
                                            const int*               ipiv,
                                            const int                strideP,
                                            double*                  B,
                                            const int                ldb,
                                            const int                strideB,
                                            int*                     info,
                                            const int                batch_count)
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

hipblasStatus_t hipblasCgetrsStridedBatched(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipblasComplex*          A,
                                            const int                lda,
                                            const int                strideA,
                                            const int*               ipiv,
                                            const int                strideP,
                                            hipblasComplex*          B,
                                            const int                ldb,
                                            const int                strideB,
                                            int*                     info,
                                            const int                batch_count)
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

hipblasStatus_t hipblasZgetrsStridedBatched(hipblasHandle_t          handle,
                                            const hipblasOperation_t trans,
                                            const int                n,
                                            const int                nrhs,
                                            hipblasDoubleComplex*    A,
                                            const int                lda,
                                            const int                strideA,
                                            const int*               ipiv,
                                            const int                strideP,
                                            hipblasDoubleComplex*    B,
                                            const int                ldb,
                                            const int                strideB,
                                            int*                     info,
                                            const int                batch_count)
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
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_sgetri_outofplace_batched(
        (rocblas_handle)handle, n, A, lda, ipiv, n, C, ldc, info, batch_count)));
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
{
    return HIPBLAS_DEMAND_ALLOC(rocBLASStatusToHIPStatus(rocsolver_dgetri_outofplace_batched(
        (rocblas_handle)handle, n, A, lda, ipiv, n, C, ldc, info, batch_count)));
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

hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t             handle,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     int*                        ipiv,
                                     hipblasDoubleComplex* const C[],
                                     const int                   ldc,
                                     int*                        info,
                                     const int                   batch_count)
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

// geqrf
hipblasStatus_t hipblasSgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              float*          A,
                              const int       lda,
                              float*          tau,
                              int*            info)
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

hipblasStatus_t hipblasDgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              double*         A,
                              const int       lda,
                              double*         tau,
                              int*            info)
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

hipblasStatus_t hipblasCgeqrf(hipblasHandle_t handle,
                              const int       m,
                              const int       n,
                              hipblasComplex* A,
                              const int       lda,
                              hipblasComplex* tau,
                              int*            info)
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

hipblasStatus_t hipblasZgeqrf(hipblasHandle_t       handle,
                              const int             m,
                              const int             n,
                              hipblasDoubleComplex* A,
                              const int             lda,
                              hipblasDoubleComplex* tau,
                              int*                  info)
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

// geqrf_batched
hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     float* const    A[],
                                     const int       lda,
                                     float* const    tau[],
                                     int*            info,
                                     const int       batch_count)
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

hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle,
                                     const int       m,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     double* const   tau[],
                                     int*            info,
                                     const int       batch_count)
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

hipblasStatus_t hipblasCgeqrfBatched(hipblasHandle_t       handle,
                                     const int             m,
                                     const int             n,
                                     hipblasComplex* const A[],
                                     const int             lda,
                                     hipblasComplex* const tau[],
                                     int*                  info,
                                     const int             batch_count)
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

hipblasStatus_t hipblasZgeqrfBatched(hipblasHandle_t             handle,
                                     const int                   m,
                                     const int                   n,
                                     hipblasDoubleComplex* const A[],
                                     const int                   lda,
                                     hipblasDoubleComplex* const tau[],
                                     int*                        info,
                                     const int                   batch_count)
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

// geqrf_strided_batched
hipblasStatus_t hipblasSgeqrfStridedBatched(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            float*          A,
                                            const int       lda,
                                            const int       strideA,
                                            float*          tau,
                                            const int       strideT,
                                            int*            info,
                                            const int       batch_count)
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

hipblasStatus_t hipblasDgeqrfStridedBatched(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            double*         A,
                                            const int       lda,
                                            const int       strideA,
                                            double*         tau,
                                            const int       strideT,
                                            int*            info,
                                            const int       batch_count)
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

hipblasStatus_t hipblasCgeqrfStridedBatched(hipblasHandle_t handle,
                                            const int       m,
                                            const int       n,
                                            hipblasComplex* A,
                                            const int       lda,
                                            const int       strideA,
                                            hipblasComplex* tau,
                                            const int       strideT,
                                            int*            info,
                                            const int       batch_count)
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

hipblasStatus_t hipblasZgeqrfStridedBatched(hipblasHandle_t       handle,
                                            const int             m,
                                            const int             n,
                                            hipblasDoubleComplex* A,
                                            const int             lda,
                                            const int             strideA,
                                            hipblasDoubleComplex* tau,
                                            const int             strideT,
                                            int*                  info,
                                            const int             batch_count)
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
{
    uint32_t solution_index = 0;
    uint32_t flags          = 0;

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
                                            int                stride_A,
                                            const void*        B,
                                            hipblasDatatype_t  b_type,
                                            int                ldb,
                                            int                stride_B,
                                            const void*        beta,
                                            void*              C,
                                            hipblasDatatype_t  c_type,
                                            int                ldc,
                                            int                stride_C,
                                            int                batch_count,
                                            hipblasDatatype_t  compute_type,
                                            hipblasGemmAlgo_t  algo)
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
                                            int                stride_A,
                                            void*              B,
                                            int                ldb,
                                            int                stride_B,
                                            int                batch_count,
                                            const void*        invA,
                                            int                invA_size,
                                            int                stride_invA,
                                            hipblasDatatype_t  compute_type)
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

} // extern "C"
