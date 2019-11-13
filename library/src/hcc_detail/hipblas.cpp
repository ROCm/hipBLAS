/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "hipblas.h"
#include "limits.h"
#include "rocblas.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    default:
        throw "Non existent OP";
    }
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
    default:
        throw "Non existent OP";
    }
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
    default:
        throw "Non existent FILL";
    }
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
    default:
        throw "Non existent FILL";
    }
}

rocblas_diagonal_ hipDiagonalToHCCDiagonal(hipblasDiagType_t diagonal)
{
    switch(diagonal)
    {
    case HIPBLAS_DIAG_NON_UNIT:
        return rocblas_diagonal_non_unit;
    case HIPBLAS_DIAG_UNIT:
        return rocblas_diagonal_unit;
    default:
        throw "Non existent DIAGONAL";
    }
}

hipblasDiagType_t HCCDiagonalToHIPDiagonal(rocblas_diagonal_ diagonal)
{
    switch(diagonal)
    {
    case rocblas_diagonal_non_unit:
        return HIPBLAS_DIAG_NON_UNIT;
    case rocblas_diagonal_unit:
        return HIPBLAS_DIAG_UNIT;
    default:
        throw "Non existent DIAGONAL";
    }
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
    default:
        throw "Non existent SIDE";
    }
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
    default:
        throw "Non existent SIDE";
    }
}

rocblas_pointer_mode HIPPointerModeToRocblasPointerMode(hipblasPointerMode_t mode)
{
    switch(mode)
    {
    case HIPBLAS_POINTER_MODE_HOST:
        return rocblas_pointer_mode_host;

    case HIPBLAS_POINTER_MODE_DEVICE:
        return rocblas_pointer_mode_device;

    default:
        throw "Non existent PointerMode";
    }
}

hipblasPointerMode_t RocblasPointerModeToHIPPointerMode(rocblas_pointer_mode mode)
{
    switch(mode)
    {
    case rocblas_pointer_mode_host:
        return HIPBLAS_POINTER_MODE_HOST;

    case rocblas_pointer_mode_device:
        return HIPBLAS_POINTER_MODE_DEVICE;

    default:
        throw "Non existent PointerMode";
    }
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

    case HIPBLAS_C_16F:
        return rocblas_datatype_f16_c;

    case HIPBLAS_C_32F:
        return rocblas_datatype_f32_c;

    case HIPBLAS_C_64F:
        return rocblas_datatype_f64_c;

    default:
        throw "Non existant DataType";
    }
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

    case rocblas_datatype_f16_c:
        return HIPBLAS_C_16F;

    case rocblas_datatype_f32_c:
        return HIPBLAS_C_32F;

    case rocblas_datatype_f64_c:
        return HIPBLAS_C_64F;

    default:
        throw "Non existant DataType";
    }
}

rocblas_gemm_algo HIPGemmAlgoToRocblasGemmAlgo(hipblasGemmAlgo_t algo)
{
    switch(algo)
    {
    case HIPBLAS_GEMM_DEFAULT:
        return rocblas_gemm_algo_standard;

    default:
        throw "Non existant GemmAlgo";
    }
}

hipblasGemmAlgo_t RocblasGemmAlgoToHIPGemmAlgo(rocblas_gemm_algo algo)
{
    switch(algo)
    {
    case rocblas_gemm_algo_standard:
        return HIPBLAS_GEMM_DEFAULT;

    default:
        throw "Non existant GemmAlgo";
    }
}

hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status_ error)
{
    switch(error)
    {
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
    default:
        throw "Unimplemented status";
    }
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
{
    int             deviceId;
    hipError_t      err;
    hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;

    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = rocBLASStatusToHIPStatus(rocblas_create_handle((rocblas_handle*)handle));
    }
    return retval;
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

// MAX
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamax((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t
    hipblasIzamax(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamax((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
}

// MIN
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamin((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamin((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasIcamin(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_icamin((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t
    hipblasIzamin(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_izamin((rocblas_handle)handle, n, (rocblas_double_complex*)x, incx, result));
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
    hipblasScasum(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_scasum((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasDzasum(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
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

hipblasStatus_t hipblasScasumBatched(hipblasHandle_t         handle,
                                     int                     n,
                                     const hipComplex* const x[],
                                     int                     incx,
                                     int                     batch_count,
                                     float*                  result)
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batch_count, result));
}

hipblasStatus_t hipblasDzasumBatched(hipblasHandle_t               handle,
                                     int                           n,
                                     const hipDoubleComplex* const x[],
                                     int                           incx,
                                     int                           batch_count,
                                     double*                       result)
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

hipblasStatus_t hipblasScasumStridedBatched(hipblasHandle_t   handle,
                                            int               n,
                                            const hipComplex* x,
                                            int               incx,
                                            int               stridex,
                                            int               batch_count,
                                            float*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_scasum_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batch_count, result));
}

hipblasStatus_t hipblasDzasumStridedBatched(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipDoubleComplex* x,
                                            int                     incx,
                                            int                     stridex,
                                            int                     batch_count,
                                            double*                 result)
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

hipblasStatus_t hipblasCaxpy(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* alpha,
                             const hipComplex* x,
                             int               incx,
                             hipComplex*       y,
                             int               incy)
{
    return rocBLASStatusToHIPStatus(rocblas_caxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)alpha,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* alpha,
                             const hipDoubleComplex* x,
                             int                     incx,
                             hipDoubleComplex*       y,
                             int                     incy)
{
    return rocBLASStatusToHIPStatus(rocblas_zaxpy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy));
}

/* not implemented
hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float
*x, int incx,  float *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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
    hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZcopy(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             hipDoubleComplex*       y,
                             int                     incy)
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

hipblasStatus_t hipblasCcopyBatched(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipComplex* const x[],
                                    int                     incx,
                                    hipComplex* const       y[],
                                    int                     incy,
                                    int                     batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_ccopy_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex**)x,
                                                          incx,
                                                          (rocblas_float_complex**)y,
                                                          incy,
                                                          batchCount));
}

hipblasStatus_t hipblasZcopyBatched(hipblasHandle_t               handle,
                                    int                           n,
                                    const hipDoubleComplex* const x[],
                                    int                           incx,
                                    hipDoubleComplex* const       y[],
                                    int                           incy,
                                    int                           batchCount)
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

hipblasStatus_t hipblasCcopyStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* x,
                                           int               incx,
                                           int               stridex,
                                           hipComplex*       y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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

hipblasStatus_t hipblasZcopyStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           hipDoubleComplex*       y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batchCount)
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

hipblasStatus_t hipblasCdotc(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* x,
                             int               incx,
                             const hipComplex* y,
                             int               incy,
                             hipComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_cdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t   handle,
                             int               n,
                             const hipComplex* x,
                             int               incx,
                             const hipComplex* y,
                             int               incy,
                             hipComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_cdotu((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy,
                                                  (rocblas_float_complex*)result));
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             const hipDoubleComplex* y,
                             int                     incy,
                             hipDoubleComplex*       result)
{
    return rocBLASStatusToHIPStatus(rocblas_zdotc((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)x,
                                                  incx,
                                                  (rocblas_double_complex*)y,
                                                  incy,
                                                  (rocblas_double_complex*)result));
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t         handle,
                             int                     n,
                             const hipDoubleComplex* x,
                             int                     incx,
                             const hipDoubleComplex* y,
                             int                     incy,
                             hipDoubleComplex*       result)
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

hipblasStatus_t hipblasCdotcBatched(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipComplex* const x[],
                                    int                     incx,
                                    const hipComplex* const y[],
                                    int                     incy,
                                    int                     batch_count,
                                    hipComplex*             result)
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

hipblasStatus_t hipblasCdotuBatched(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipComplex* const x[],
                                    int                     incx,
                                    const hipComplex* const y[],
                                    int                     incy,
                                    int                     batch_count,
                                    hipComplex*             result)
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

hipblasStatus_t hipblasZdotcBatched(hipblasHandle_t               handle,
                                    int                           n,
                                    const hipDoubleComplex* const x[],
                                    int                           incx,
                                    const hipDoubleComplex* const y[],
                                    int                           incy,
                                    int                           batch_count,
                                    hipDoubleComplex*             result)
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

hipblasStatus_t hipblasZdotuBatched(hipblasHandle_t               handle,
                                    int                           n,
                                    const hipDoubleComplex* const x[],
                                    int                           incx,
                                    const hipDoubleComplex* const y[],
                                    int                           incy,
                                    int                           batch_count,
                                    hipDoubleComplex*             result)
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

hipblasStatus_t hipblasCdotcStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* x,
                                           int               incx,
                                           int               stridex,
                                           const hipComplex* y,
                                           int               incy,
                                           int               stridey,
                                           int               batch_count,
                                           hipComplex*       result)
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

hipblasStatus_t hipblasCdotuStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* x,
                                           int               incx,
                                           int               stridex,
                                           const hipComplex* y,
                                           int               incy,
                                           int               stridey,
                                           int               batch_count,
                                           hipComplex*       result)
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

hipblasStatus_t hipblasZdotcStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           const hipDoubleComplex* y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batch_count,
                                           hipDoubleComplex*       result)
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

hipblasStatus_t hipblasZdotuStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* x,
                                           int                     incx,
                                           int                     stridex,
                                           const hipDoubleComplex* y,
                                           int                     incy,
                                           int                     stridey,
                                           int                     batch_count,
                                           hipDoubleComplex*       result)
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
    hipblasScnrm2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(
        rocblas_scnrm2((rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, result));
}

hipblasStatus_t hipblasDznrm2(
    hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result)
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

hipblasStatus_t hipblasScnrm2Batched(hipblasHandle_t         handle,
                                     int                     n,
                                     const hipComplex* const x[],
                                     int                     incx,
                                     int                     batchCount,
                                     float*                  result)
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex* const*)x, incx, batchCount, result));
}

hipblasStatus_t hipblasDznrm2Batched(hipblasHandle_t               handle,
                                     int                           n,
                                     const hipDoubleComplex* const x[],
                                     int                           incx,
                                     int                           batchCount,
                                     double*                       result)
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

hipblasStatus_t hipblasScnrm2StridedBatched(hipblasHandle_t   handle,
                                            int               n,
                                            const hipComplex* x,
                                            int               incx,
                                            int               stridex,
                                            int               batchCount,
                                            float*            result)
{
    return rocBLASStatusToHIPStatus(rocblas_scnrm2_strided_batched(
        (rocblas_handle)handle, n, (rocblas_float_complex*)x, incx, stridex, batchCount, result));
}

hipblasStatus_t hipblasDznrm2StridedBatched(hipblasHandle_t         handle,
                                            int                     n,
                                            const hipDoubleComplex* x,
                                            int                     incx,
                                            int                     stridex,
                                            int                     batchCount,
                                            double*                 result)
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

hipblasStatus_t hipblasCrot(hipblasHandle_t   handle,
                            int               n,
                            hipComplex*       x,
                            int               incx,
                            hipComplex*       y,
                            int               incy,
                            const float*      c,
                            const hipComplex* s)
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
                             hipComplex*     x,
                             int             incx,
                             hipComplex*     y,
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

hipblasStatus_t hipblasZrot(hipblasHandle_t         handle,
                            int                     n,
                            hipDoubleComplex*       x,
                            int                     incx,
                            hipDoubleComplex*       y,
                            int                     incy,
                            const double*           c,
                            const hipDoubleComplex* s)
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

hipblasStatus_t hipblasZdrot(hipblasHandle_t   handle,
                             int               n,
                             hipDoubleComplex* x,
                             int               incx,
                             hipDoubleComplex* y,
                             int               incy,
                             const double*     c,
                             const double*     s)
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

hipblasStatus_t hipblasCrotBatched(hipblasHandle_t   handle,
                                   int               n,
                                   hipComplex* const x[],
                                   int               incx,
                                   hipComplex* const y[],
                                   int               incy,
                                   const float*      c,
                                   const hipComplex* s,
                                   int               batchCount)
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

hipblasStatus_t hipblasCsrotBatched(hipblasHandle_t   handle,
                                    int               n,
                                    hipComplex* const x[],
                                    int               incx,
                                    hipComplex* const y[],
                                    int               incy,
                                    const float*      c,
                                    const float*      s,
                                    int               batchCount)
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

hipblasStatus_t hipblasZrotBatched(hipblasHandle_t         handle,
                                   int                     n,
                                   hipDoubleComplex* const x[],
                                   int                     incx,
                                   hipDoubleComplex* const y[],
                                   int                     incy,
                                   const double*           c,
                                   const hipDoubleComplex* s,
                                   int                     batchCount)
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

hipblasStatus_t hipblasZdrotBatched(hipblasHandle_t         handle,
                                    int                     n,
                                    hipDoubleComplex* const x[],
                                    int                     incx,
                                    hipDoubleComplex* const y[],
                                    int                     incy,
                                    const double*           c,
                                    const double*           s,
                                    int                     batchCount)
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

hipblasStatus_t hipblasCrotStridedBatched(hipblasHandle_t   handle,
                                          int               n,
                                          hipComplex*       x,
                                          int               incx,
                                          int               stridex,
                                          hipComplex*       y,
                                          int               incy,
                                          int               stridey,
                                          const float*      c,
                                          const hipComplex* s,
                                          int               batchCount)
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
                                           hipComplex*     x,
                                           int             incx,
                                           int             stridex,
                                           hipComplex*     y,
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

hipblasStatus_t hipblasZrotStridedBatched(hipblasHandle_t         handle,
                                          int                     n,
                                          hipDoubleComplex*       x,
                                          int                     incx,
                                          int                     stridex,
                                          hipDoubleComplex*       y,
                                          int                     incy,
                                          int                     stridey,
                                          const double*           c,
                                          const hipDoubleComplex* s,
                                          int                     batchCount)
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

hipblasStatus_t hipblasZdrotStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           hipDoubleComplex* x,
                                           int               incx,
                                           int               stridex,
                                           hipDoubleComplex* y,
                                           int               incy,
                                           int               stridey,
                                           const double*     c,
                                           const double*     s,
                                           int               batchCount)
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

hipblasStatus_t
    hipblasCrotg(hipblasHandle_t handle, hipComplex* a, hipComplex* b, float* c, hipComplex* s)
{
    return rocBLASStatusToHIPStatus(rocblas_crotg((rocblas_handle)handle,
                                                  (rocblas_float_complex*)a,
                                                  (rocblas_float_complex*)b,
                                                  c,
                                                  (rocblas_float_complex*)s));
}

hipblasStatus_t hipblasZrotg(hipblasHandle_t   handle,
                             hipDoubleComplex* a,
                             hipDoubleComplex* b,
                             double*           c,
                             hipDoubleComplex* s)
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

hipblasStatus_t hipblasCrotgBatched(hipblasHandle_t   handle,
                                    hipComplex* const a[],
                                    hipComplex* const b[],
                                    float* const      c[],
                                    hipComplex* const s[],
                                    int               batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_crotg_batched((rocblas_handle)handle,
                                                          (rocblas_float_complex**)a,
                                                          (rocblas_float_complex**)b,
                                                          c,
                                                          (rocblas_float_complex**)s,
                                                          batchCount));
}

hipblasStatus_t hipblasZrotgBatched(hipblasHandle_t         handle,
                                    hipDoubleComplex* const a[],
                                    hipDoubleComplex* const b[],
                                    double* const           c[],
                                    hipDoubleComplex* const s[],
                                    int                     batchCount)
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
                                           hipComplex*     a,
                                           int             stride_a,
                                           hipComplex*     b,
                                           int             stride_b,
                                           float*          c,
                                           int             stride_c,
                                           hipComplex*     s,
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

hipblasStatus_t hipblasZrotgStridedBatched(hipblasHandle_t   handle,
                                           hipDoubleComplex* a,
                                           int               stride_a,
                                           hipDoubleComplex* b,
                                           int               stride_b,
                                           double*           c,
                                           int               stride_c,
                                           hipDoubleComplex* s,
                                           int               stride_s,
                                           int               batchCount)
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

hipblasStatus_t
    hipblasCscal(hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal(
        (rocblas_handle)handle, n, (rocblas_float_complex*)alpha, (rocblas_float_complex*)x, incx));
}

hipblasStatus_t
    hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(
        rocblas_csscal((rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx));
}

hipblasStatus_t hipblasZscal(
    hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_zscal((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_double_complex*)alpha,
                                                  (rocblas_double_complex*)x,
                                                  incx));
}

hipblasStatus_t
    hipblasZdscal(hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx)
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

hipblasStatus_t hipblasCscalBatched(hipblasHandle_t   handle,
                                    int               n,
                                    const hipComplex* alpha,
                                    hipComplex* const x[],
                                    int               incx,
                                    int               batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_float_complex*)alpha,
                                                          (rocblas_float_complex* const*)x,
                                                          incx,
                                                          batchCount));
}

hipblasStatus_t hipblasZscalBatched(hipblasHandle_t         handle,
                                    int                     n,
                                    const hipDoubleComplex* alpha,
                                    hipDoubleComplex* const x[],
                                    int                     incx,
                                    int                     batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_zscal_batched((rocblas_handle)handle,
                                                          n,
                                                          (rocblas_double_complex*)alpha,
                                                          (rocblas_double_complex* const*)x,
                                                          incx,
                                                          batchCount));
}

hipblasStatus_t hipblasCsscalBatched(hipblasHandle_t   handle,
                                     int               n,
                                     const float*      alpha,
                                     hipComplex* const x[],
                                     int               incx,
                                     int               batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex* const*)x, incx, batchCount));
}

hipblasStatus_t hipblasZdscalBatched(hipblasHandle_t         handle,
                                     int                     n,
                                     const double*           alpha,
                                     hipDoubleComplex* const x[],
                                     int                     incx,
                                     int                     batchCount)
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

hipblasStatus_t hipblasCscalStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           const hipComplex* alpha,
                                           hipComplex*       x,
                                           int               incx,
                                           int               stridex,
                                           int               batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_cscal_strided_batched((rocblas_handle)handle,
                                                                  n,
                                                                  (rocblas_float_complex*)alpha,
                                                                  (rocblas_float_complex*)x,
                                                                  incx,
                                                                  stridex,
                                                                  batchCount));
}

hipblasStatus_t hipblasZscalStridedBatched(hipblasHandle_t         handle,
                                           int                     n,
                                           const hipDoubleComplex* alpha,
                                           hipDoubleComplex*       x,
                                           int                     incx,
                                           int                     stridex,
                                           int                     batchCount)
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
                                            hipComplex*     x,
                                            int             incx,
                                            int             stridex,
                                            int             batchCount)
{
    return rocBLASStatusToHIPStatus(rocblas_csscal_strided_batched(
        (rocblas_handle)handle, n, alpha, (rocblas_float_complex*)x, incx, stridex, batchCount));
}

hipblasStatus_t hipblasZdscalStridedBatched(hipblasHandle_t   handle,
                                            int               n,
                                            const double*     alpha,
                                            hipDoubleComplex* x,
                                            int               incx,
                                            int               stridex,
                                            int               batchCount)
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

hipblasStatus_t
    hipblasCswap(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy)
{
    return rocBLASStatusToHIPStatus(rocblas_cswap((rocblas_handle)handle,
                                                  n,
                                                  (rocblas_float_complex*)x,
                                                  incx,
                                                  (rocblas_float_complex*)y,
                                                  incy));
}

hipblasStatus_t hipblasZswap(
    hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy)
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
                                    hipComplex*     x[],
                                    int             incx,
                                    hipComplex*     y[],
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

hipblasStatus_t hipblasZswapBatched(hipblasHandle_t   handle,
                                    int               n,
                                    hipDoubleComplex* x[],
                                    int               incx,
                                    hipDoubleComplex* y[],
                                    int               incy,
                                    int               batchCount)
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
                                           hipComplex*     x,
                                           int             incx,
                                           int             stridex,
                                           hipComplex*     y,
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

hipblasStatus_t hipblasZswapStridedBatched(hipblasHandle_t   handle,
                                           int               n,
                                           hipDoubleComplex* x,
                                           int               incx,
                                           int               stridex,
                                           hipDoubleComplex* y,
                                           int               incy,
                                           int               stridey,
                                           int               batchCount)
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
    return rocBLASStatusToHIPStatus(rocblas_strsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
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
    return rocBLASStatusToHIPStatus(rocblas_dtrsv((rocblas_handle)handle,
                                                  (rocblas_fill)uplo,
                                                  hipOperationToHCCOperation(transA),
                                                  hipDiagonalToHCCDiagonal(diag),
                                                  m,
                                                  A,
                                                  lda,
                                                  x,
                                                  incx));
}

//------------------------------------------------------------------------------------------------------------

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
    return rocBLASStatusToHIPStatus(rocblas_strsm((rocblas_handle)handle,
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
    return rocBLASStatusToHIPStatus(rocblas_dtrsm((rocblas_handle)handle,
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

hipblasStatus_t hipblasZgemm(hipblasHandle_t         handle,
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

hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t         handle,
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

hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t               handle,
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

hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t    handle,
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

hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t         handle,
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

#ifdef __cplusplus
}
#endif

extern "C" hipblasStatus_t hipblasGemmEx(hipblasHandle_t    handle,
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

    uint32_t flags = 0;

    size_t* workspace_size = 0;

    void* workspace = 0;

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
                                                    flags,
                                                    workspace_size,
                                                    workspace));
}
