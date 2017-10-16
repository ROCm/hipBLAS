/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas.h"
#include "hipblas.h"
#include "limits.h"

#ifdef __cplusplus
extern "C" {
#endif

rocblas_operation_ hipOperationToHCCOperation( hipblasOperation_t op) {
        switch (op)
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

hipblasOperation_t HCCOperationToHIPOperation( rocblas_operation_ op) {
	switch (op)
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

rocblas_pointer_mode HIPPointerModeToRocblasPointerMode( hipblasPointerMode_t mode)
{
    switch (mode)
    {
        case HIPBLAS_POINTER_MODE_HOST :
            return rocblas_pointer_mode_host;

        case HIPBLAS_POINTER_MODE_DEVICE :
            return rocblas_pointer_mode_device;

        default:
            throw "Non existent PointerMode";
    }
}


hipblasPointerMode_t RocblasPointerModeToHIPPointerMode( rocblas_pointer_mode mode)
{
    switch (mode)
    {
        case rocblas_pointer_mode_host :
            return HIPBLAS_POINTER_MODE_HOST;

        case rocblas_pointer_mode_device :
            return HIPBLAS_POINTER_MODE_DEVICE;

        default:
            throw "Non existent PointerMode";
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


hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
  int deviceId;
  hipError_t err;
  hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;

  if (handle == nullptr)
  {
     handle = (hipblasHandle_t *) new rocblas_handle();
  }

  err = hipGetDevice(&deviceId);
  if (err == hipSuccess) {
      retval = rocBLASStatusToHIPStatus (rocblas_create_handle((rocblas_handle *) handle));
  }
  return retval;
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    return rocBLASStatusToHIPStatus(rocblas_destroy_handle((rocblas_handle) handle));
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId) {
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  }
  return rocBLASStatusToHIPStatus(rocblas_set_stream((rocblas_handle) handle, streamId));
}

hipblasStatus_t  hipblasGetStream(hipblasHandle_t handle, hipStream_t *streamId) {
  if (handle == nullptr) {
    return HIPBLAS_STATUS_NOT_INITIALIZED;
  }
  return rocBLASStatusToHIPStatus(rocblas_get_stream((rocblas_handle) handle, streamId));
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode){
    return rocBLASStatusToHIPStatus(rocblas_set_pointer_mode((rocblas_handle) handle, HIPPointerModeToRocblasPointerMode(mode)));
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t *mode){
    rocblas_pointer_mode rocblas_mode;
    rocblas_status status = rocblas_get_pointer_mode((rocblas_handle) handle, &rocblas_mode);
    *mode = RocblasPointerModeToHIPPointerMode(rocblas_mode);
    return rocBLASStatusToHIPStatus(status);
}

hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
    return rocBLASStatusToHIPStatus(rocblas_set_vector(n, elemSize, x, incx, y, incy));
}

hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy){
    return rocBLASStatusToHIPStatus(rocblas_get_vector(n, elemSize, x, incx, y, incy));
}

hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
    return rocBLASStatusToHIPStatus(rocblas_set_matrix(rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb){
    return rocBLASStatusToHIPStatus(rocblas_get_matrix(rows, cols, elemSize, A, lda, B, ldb));
}

hipblasStatus_t hipblasSgeam(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
   // TODO: Remove const_cast
	    return rocBLASStatusToHIPStatus(rocblas_sgeam((rocblas_handle)handle, hipOperationToHCCOperation(transa), hipOperationToHCCOperation(transb), m, n, alpha, const_cast<float*>(A), lda, const_cast<float*>(B), ldb, beta, C, ldc));
}

hipblasStatus_t hipblasDgeam(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
   // TODO: Remove const_cast
	    return rocBLASStatusToHIPStatus(rocblas_dgeam((rocblas_handle)handle, hipOperationToHCCOperation(transa), hipOperationToHCCOperation(transb), m, n, alpha, const_cast<double*>(A), lda, const_cast<double*>(B), ldb, beta, C, ldc));
}

hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, const float *x, int incx, float  *result){
	return rocBLASStatusToHIPStatus(rocblas_sasum((rocblas_handle)handle, n, const_cast<float*>(x), incx, result));
}

hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, const double *x, int incx, double *result){
	return rocBLASStatusToHIPStatus(rocblas_dasum((rocblas_handle)handle, n, const_cast<double*>(x), incx, result));
}

/* not implemented
hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy) {
	return rocBLASStatusToHIPStatus(rocblas_saxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double *alpha,   const double *x, int incx, double *y, int incy) {
	return rocBLASStatusToHIPStatus(rocblas_daxpy((rocblas_handle)handle, n, alpha, x, incx, y, incy));
}

/* not implemented
hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
	return rocBLASStatusToHIPStatus(rocblas_scopy((rocblas_handle)handle, n, x, incx, y, incy));
}

hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
	return rocBLASStatusToHIPStatus(rocblas_dcopy((rocblas_handle)handle, n, x, incx, y, incy));
}

/* not implemented
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/


hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){
	return rocBLASStatusToHIPStatus(rocblas_sdot((rocblas_handle)handle, n, x, incx, y, incy, result));
}

hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){
	return rocBLASStatusToHIPStatus(rocblas_ddot((rocblas_handle)handle, n, x, incx, y, incy, result));
}

/* not implemented
hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx){
	return rocBLASStatusToHIPStatus(rocblas_sscal((rocblas_handle)handle, n, alpha, x, incx));
}

hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx){
	return rocBLASStatusToHIPStatus(rocblas_dscal((rocblas_handle)handle, n, alpha, x, incx));
}

/*   complex not implemented
hipblasStatus_t  hipblasCscal(hipblasHandle_t handle, int n, const hipComplex *alpha,  hipComplex *x, int incx){
	return rocBLASStatusToHIPStatus(rocblas_cscal((rocblas_handle)handle, n, (const rocblas_precision_complex_single*)alpha,  (rocblas_precision_complex_single*)x, incx));
}

hipblasStatus_t  hipblasZscal(hipblasHandle_t handle, int n, const hipDoubleComplex *alpha,  hipDoubleComplex *x, int incx){
	return rocBLASStatusToHIPStatus(rocblas_zscal((rocblas_handle)handle, n, (const rocblas_precision_complex_double*)alpha,  (rocblas_precision_complex_double*)x, incx));
}
*/

/* not implemented
hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}

hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                           const float *x, int incx,  const float *beta,  float *y, int incy){
        // TODO: Remove const_cast
	return rocBLASStatusToHIPStatus(rocblas_sgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n, alpha, const_cast<float*>(A), lda, const_cast<float*>(x), incx, beta, y, incy));
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
                           const double *x, int incx,  const double *beta,  double *y, int incy){
        // TODO: Remove const_cast
	return rocBLASStatusToHIPStatus(rocblas_dgemv((rocblas_handle)handle, hipOperationToHCCOperation(trans), m, n, alpha, const_cast<double*>(A), lda, const_cast<double*>(x), incx, beta, y, incy));
}

/* not implemented
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
	return rocBLASStatusToHIPStatus(rocblas_sger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

hipblasStatus_t  hipblasDger(hipblasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda){
	return rocBLASStatusToHIPStatus(rocblas_dger((rocblas_handle)handle, m, n, alpha, x, incx, y, incy, A, lda));
}

/* not implemented
hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount){return HIPBLAS_STATUS_NOT_SUPPORTED;}
*/

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc){
   // TODO: Remove const_cast
	return rocBLASStatusToHIPStatus(rocblas_sgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<float*>(A),  lda, const_cast<float*>(B),  ldb, beta, C,  ldc));
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc){
	return rocBLASStatusToHIPStatus(rocblas_dgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, alpha, const_cast<double*>(A),  lda, const_cast<double*>(B),  ldb, beta, C,  ldc));
}

/*   complex not implemented
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A, int lda, const hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc){
	return rocBLASStatusToHIPStatus(rocblas_cgemm((rocblas_handle) handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m,  n,  k, (const rocblas_precision_complex_single*)(alpha), const_cast<rocblas_precision_complex_single*>((const rocblas_precision_complex_single*)(A)),  lda, const_cast<rocblas_precision_complex_single*>((const rocblas_precision_complex_single*)(B)),  ldb, (const rocblas_precision_complex_single*)(beta), (rocblas_precision_complex_single*)(C),  ldc));
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda, const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *C, int ldc){
	return rocBLASStatusToHIPStatus(rocblas_zgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m, n, k, (const rocblas_precision_complex_double*)(alpha), const_cast<rocblas_precision_complex_double*>((const rocblas_precision_complex_double*)(A)),  lda, const_cast<rocblas_precision_complex_double*>((const rocblas_precision_complex_double*)(B)),  ldb, (const rocblas_precision_complex_double*)(beta), (rocblas_precision_complex_double*)(C),  ldc));
}

hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const __half *alpha, __half *A, int lda, __half *B, int ldb, const __half *beta, __half *C, int ldc){
	return rocBLASStatusToHIPStatus(rocblas_hgemm((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb), m, n, k, alpha, A, lda, rocblas_half *B, ldb, beta, rocblas_half *C, ldc));
}
*/

hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
int m, int n, int k,  const float *alpha, const float *A, int lda, long long bsa, const float *B, int ldb, long long bsb, const float *beta, float *C, int ldc, long long bsc, int batchCount)
{
    int bsa_int, bsb_int, bsc_int;
    if (bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
    {
        bsa_int = static_cast<int>(bsa);
        bsb_int = static_cast<int>(bsb);
        bsc_int = static_cast<int>(bsc);
    }
    else
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    return rocBLASStatusToHIPStatus(rocblas_sgemm_strided_batched((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb),
    m, n, k, alpha, const_cast<float*>(A), lda, bsa_int, const_cast<float*>(B), ldb, bsb_int, beta, C, ldc, bsc_int, batchCount));
}

hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
int m, int n, int k,  const double *alpha, const double *A, int lda, long long bsa, const double *B, int ldb, long long bsb, const double *beta, double *C, int ldc, long long bsc, int batchCount)
{
    int bsa_int, bsb_int, bsc_int;
    if (bsa < INT_MAX && bsb < INT_MAX && bsc < INT_MAX)
    {
        bsa_int = static_cast<int>(bsa);
        bsb_int = static_cast<int>(bsb);
        bsc_int = static_cast<int>(bsc);
    }
    else
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    return rocBLASStatusToHIPStatus(rocblas_dgemm_strided_batched((rocblas_handle)handle, hipOperationToHCCOperation(transa),  hipOperationToHCCOperation(transb),
    m, n, k, alpha, const_cast<double*>(A), lda, bsa_int, const_cast<double*>(B), ldb, bsb_int, beta, C, ldc, bsc_int, batchCount));
}

#ifdef __cplusplus
}
#endif
