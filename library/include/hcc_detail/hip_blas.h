/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include <hip/hip_runtime_api.h>
// #include <hip/hip_complex.h>
// #include <hip/hip_fp16.h>

//HGSOS for Kalmar leave it as C++, only cublas needs C linkage.


#ifdef __cplusplus
extern "C" {
#endif

/*
// Forward declarations
enum rocblas_status: unsigned short;
enum rocblas_operation : unsigned short;

typedef struct Hcblaslibrary* hcblasHandle_t;
typedef hcblasHandle_t hipblasHandle_t;
*/

typedef rocblas_handle hipblasHandle_t;

rocblas_operation hipOperationToHCCOperation( hipblasOperation_t op);

hipblasOperation_t HCCOperationToHIPOperation( rocblas_operation op);

hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status rocBlasStatus);

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);

hipblasStatus_t  hipblasGetStream(hipblasHandle_t handle, hipStream_t *streamId);

// not implemented
hipblasStatus_t hipblasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);

// not implemented
hipblasStatus_t hipblasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy);

// not implemented
hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

// not implemented
hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);

hipblasStatus_t  hipblasSasum(hipblasHandle_t handle, int n, const float *x, int incx, float  *result);

hipblasStatus_t  hipblasDasum(hipblasHandle_t handle, int n, const double *x, int incx, double *result);

// not implemented
hipblasStatus_t  hipblasSasumBatched(hipblasHandle_t handle, int n, float *x, int incx, float  *result, int batchCount);

// not implemented
hipblasStatus_t  hipblasDasumBatched(hipblasHandle_t handle, int n, double *x, int incx, double *result, int batchCount);

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float *alpha,   const float *x, int incx, float *y, int incy);

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double *alpha,   const double *x, int incx, double *y, int incy) ;

// not implemented
hipblasStatus_t hipblasSaxpyBatched(hipblasHandle_t handle, int n, const float *alpha, const float *x, int incx,  float *y, int incy, int batchCount);

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy);

hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy);

// not implemented
hipblasStatus_t hipblasScopyBatched(hipblasHandle_t handle, int n, const float *x, int incx, float *y, int incy, int batchCount);

// not implemented
hipblasStatus_t hipblasDcopyBatched(hipblasHandle_t handle, int n, const double *x, int incx, double *y, int incy, int batchCount);


hipblasStatus_t hipblasSdot (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result);

hipblasStatus_t hipblasDdot (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);

// not implemented
hipblasStatus_t hipblasSdotBatched (hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result, int batchCount);

// not implemented
hipblasStatus_t hipblasDdotBatched (hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result, int batchCount);

hipblasStatus_t  hipblasSscal(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx);

hipblasStatus_t  hipblasDscal(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx);

/* not implementes, requires complex support
hipblasStatus_t  hipblasCscal(hipblasHandle_t handle, int n, const hipComplex *alpha,  hipComplex *x, int incx);

hipblasStatus_t  hipblasZscal(hipblasHandle_t handle, int n, const hipDoubleComplex *alpha,  hipDoubleComplex *x, int incx);
*/

// not implemented
hipblasStatus_t  hipblasSscalBatched(hipblasHandle_t handle, int n, const float *alpha,  float *x, int incx, int batchCount);

// not implemented
hipblasStatus_t  hipblasDscalBatched(hipblasHandle_t handle, int n, const double *alpha,  double *x, int incx, int batchCount);

hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                           const float *x, int incx,  const float *beta,  float *y, int incy);

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda,
                           const double *x, int incx,  const double *beta,  double *y, int incy);

// not implemented
hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float *alpha, float *A, int lda,
                           float *x, int incx,  const float *beta,  float *y, int incy, int batchCount);

hipblasStatus_t  hipblasSger(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);

hipblasStatus_t  hipblasDger(hipblasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);

// not implemented
hipblasStatus_t  hipblasSgerBatched(hipblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda, int batchCount);

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);

/* not implementes, requires complex support
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A, int lda, const hipComplex *B, int ldb, const hipComplex *beta, hipComplex *C, int ldc);

hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda, const hipDoubleComplex *B, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *C, int ldc);

hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const __half *alpha, __half *A, int lda, __half *B, int ldb, const __half *beta, __half *C, int ldc);
*/

// not implemented
hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const float *alpha, const float *A[], int lda, const float *B[], int ldb, const float *beta, float *C[], int ldc, int batchCount);

// not implemented
hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const double *alpha, const double *A[], int lda, const double *B[], int ldb, const double *beta, double *C[], int ldc, int batchCount);


/* not implementes, requires complex support
hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle,  hipblasOperation_t transa, hipblasOperation_t transb,
                           int m, int n, int k,  const hipComplex *alpha, const hipComplex *A[], int lda, const hipComplex *B[], int ldb, const hipComplex *beta, hipComplex *C[], int ldc, int batchCount);

*/


#ifdef __cplusplus
}
#endif

