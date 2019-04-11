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

hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_isamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return rocBLASStatusToHIPStatus(rocblas_idamax((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(rocblas_sasum((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(rocblas_dasum((rocblas_handle)handle, n, x, incx, result));
}

/* not implemented
hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float
*result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double
*result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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

/* not implemented
hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float
*x, int incx,  float *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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

/* not implemented
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float
*y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double
*y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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

/* not implemented
hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const
float *y, int incy, float *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const
double *y, int incy, double *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{
    return rocBLASStatusToHIPStatus(rocblas_snrm2((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t
    hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{
    return rocBLASStatusToHIPStatus(rocblas_dnrm2((rocblas_handle)handle, n, x, incx, result));
}

hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_sscal((rocblas_handle)handle, n, alpha, x, incx));
}

hipblasStatus_t
    hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
{
    return rocBLASStatusToHIPStatus(rocblas_dscal((rocblas_handle)handle, n, alpha, x, incx));
}

/*   complex not implemented
hipblasStatus_t  hipblasCscal(hipblasHandle_t handle, int n, const hipComplex *alpha,  hipComplex
*x, int incx){
        return rocBLASStatusToHIPStatus(rocblas_cscal((rocblas_handle)handle, n, (const
rocblas_precision_complex_single*)alpha,  (rocblas_precision_complex_single*)x, incx));
}

hipblasStatus_t  hipblasZscal(hipblasHandle_t handle, int n, const hipDoubleComplex *alpha,
hipDoubleComplex *x, int incx){
        return rocBLASStatusToHIPStatus(rocblas_zscal((rocblas_handle)handle, n, (const
rocblas_precision_complex_double*)alpha,  (rocblas_precision_complex_double*)x, incx));
}
*/

/* not implemented
hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x,
int incx, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x,
int incx, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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

/* not implemented
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n,
const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int
batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

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

/* not implemented
hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const
float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount){return
HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

//------------------------------------------------------------------------------------------------------------

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

/*   complex not implemented
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t
transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A, int
lda, const hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
        return rocBLASStatusToHIPStatus(rocblas_cgemm((rocblas_handle) handle,
hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, (const
rocblas_precision_complex_single*)(alpha), const_cast<rocblas_precision_complex_single*>((const
rocblas_precision_complex_single*)(A)),  lda, const_cast<rocblas_precision_complex_single*>((const
rocblas_precision_complex_single*)(B)),  ldb, (const rocblas_precision_complex_single*)(beta),
(rocblas_precision_complex_single*)(C),  ldc));
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t
transb,
                           int m, int n, int k,  const hipDoubleComplex *alpha, const
hipDoubleComplex *A, int lda, const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta,
hipDoubleComplex *C, int ldc){
        return rocBLASStatusToHIPStatus(rocblas_zgemm((rocblas_handle)handle,
hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m, n, k, (const
rocblas_precision_complex_double*)(alpha), const_cast<rocblas_precision_complex_double*>((const
rocblas_precision_complex_double*)(A)),  lda, const_cast<rocblas_precision_complex_double*>((const
rocblas_precision_complex_double*)(B)),  ldb, (const rocblas_precision_complex_double*)(beta),
(rocblas_precision_complex_double*)(C),  ldc));
}
*/

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

#ifdef __cplusplus
}
#endif

template <typename T>
rocblas_status gemm_template(rocblas_handle    handle,
                             rocblas_operation transA,
                             rocblas_operation transB,
                             rocblas_int       m,
                             rocblas_int       n,
                             rocblas_int       k,
                             const T*          alpha,
                             const T*          A,
                             rocblas_int       lda,
                             const T*          B,
                             rocblas_int       ldb,
                             const T*          beta,
                             T*                C,
                             rocblas_int       ldc);

template <>
rocblas_status gemm_template<float>(rocblas_handle    handle,
                                    rocblas_operation transA,
                                    rocblas_operation transB,
                                    rocblas_int       M,
                                    rocblas_int       N,
                                    rocblas_int       K,
                                    const float*      alpha,
                                    const float*      A,
                                    rocblas_int       lda,
                                    const float*      B,
                                    rocblas_int       ldb,
                                    const float*      beta,
                                    float*            C,
                                    rocblas_int       ldc)
{
    return rocblas_sgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
rocblas_status gemm_template<double>(rocblas_handle    handle,
                                     rocblas_operation transA,
                                     rocblas_operation transB,
                                     rocblas_int       M,
                                     rocblas_int       N,
                                     rocblas_int       K,
                                     const double*     alpha,
                                     const double*     A,
                                     rocblas_int       lda,
                                     const double*     B,
                                     rocblas_int       ldb,
                                     const double*     beta,
                                     double*           C,
                                     rocblas_int       ldc)
{
    return rocblas_dgemm(handle, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename T>
hipblasStatus_t hipblasGemmBatched_template(hipblasHandle_t    handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int                m,
                                            int                n,
                                            int                k,
                                            const T*           alpha,
                                            const T* const     A[],
                                            int                lda,
                                            const T* const     B[],
                                            int                ldb,
                                            const T*           beta,
                                            T* const           C[],
                                            int                ldc,
                                            int                batchCount)
{
    if(batchCount < 0 || m < 0 || n < 0 || k < 0 || lda < 0 || ldb < 0 || ldc < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    T *hA[batchCount], *hB[batchCount], *hC[batchCount];
    T  h_alpha, h_beta;
    // copy arrays from device to host
    hipError_t err_A = hipMemcpy(hA, A, batchCount * sizeof(*A), hipMemcpyDeviceToHost);
    hipError_t err_B = hipMemcpy(hB, B, batchCount * sizeof(*B), hipMemcpyDeviceToHost);
    hipError_t err_C = hipMemcpy(hC, C, batchCount * sizeof(*C), hipMemcpyDeviceToHost);
    if((err_A != hipSuccess) || (err_B != hipSuccess) || (err_C != hipSuccess))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // copy alpha and beta to host if they are on device
    if(rocblas_pointer_to_mode((void*)alpha) == rocblas_pointer_mode_device)
    {
        hipError_t err_alpha = hipMemcpy(&h_alpha, alpha, sizeof(T), hipMemcpyDeviceToHost);
        hipError_t err_beta  = hipMemcpy(&h_beta, beta, sizeof(T), hipMemcpyDeviceToHost);
        if((err_alpha != hipSuccess) || (err_beta != hipSuccess))
        {
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
    }
    else
    {
        h_alpha = *alpha;
        h_beta  = *beta;
    }
    for(int i = 0; i < batchCount; i++)
    {
        rocblas_status status = gemm_template<T>((rocblas_handle)handle,
                                                 hipOperationToHCCOperation(transa),
                                                 hipOperationToHCCOperation(transb),
                                                 m,
                                                 n,
                                                 k,
                                                 &h_alpha,
                                                 const_cast<T*>(hA[i]),
                                                 lda,
                                                 const_cast<T*>(hB[i]),
                                                 ldb,
                                                 &h_beta,
                                                 hC[i],
                                                 ldc);
        if(status != rocblas_status_success)
        {
            return rocBLASStatusToHIPStatus(status);
        }
    }
    return HIPBLAS_STATUS_SUCCESS;
}

extern "C" hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t    handle,
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
                                               int                batchCount)
{
    return hipblasGemmBatched_template<float>(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

extern "C" hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t    handle,
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
                                               int                batchCount)
{
    return hipblasGemmBatched_template<double>(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
}

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
