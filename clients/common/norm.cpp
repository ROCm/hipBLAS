/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "norm.h"
#include "cblas.h"
#include "hipblas.h"
#include <stdio.h>

/* =====================================================================
     README: Norm check: norm(A-B)/norm(A), evaluate relative error
             Numerically, it is recommended by lapack.

    Call lapack fortran routines that do not exsit in cblas library.
    No special header is required. But need to declare
    function prototype

    All the functions are fortran and should append underscore (_) while declaring prototype and
   calling.
    xlange and xaxpy prototype are like following
    =================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

float  slange_(char* norm_type, int* m, int* n, float* A, int* lda, float* work);
double dlange_(char* norm_type, int* m, int* n, double* A, int* lda, double* work);
float  clange_(char* norm_type, int* m, int* n, hipblasComplex* A, int* lda, float* work);
double zlange_(char* norm_type, int* m, int* n, hipblasDoubleComplex* A, int* lda, double* work);

float  slansy_(char* norm_type, char* uplo, int* n, float* A, int* lda, float* work);
double dlansy_(char* norm_type, char* uplo, int* n, double* A, int* lda, double* work);
//  float  clanhe_(char* norm_type, char* uplo, int* n, hipblasComplex* A, int* lda, float* work);
//  double zlanhe_(char* norm_type, char* uplo, int* n, hipblasDoubleComplex* A, int* lda, double*
//  work);

void saxpy_(int* n, float* alpha, float* x, int* incx, float* y, int* incy);
void daxpy_(int* n, double* alpha, double* x, int* incx, double* y, int* incy);
void caxpy_(int* n, float* alpha, hipblasComplex* x, int* incx, hipblasComplex* y, int* incy);
void zaxpy_(
    int* n, double* alpha, hipblasDoubleComplex* x, int* incx, hipblasDoubleComplex* y, int* incy);

#ifdef __cplusplus
}
#endif

/* ============================Norm Check for General Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two matrices hCPU & hGPU */
template <>
double norm_check_general<float>(char norm_type, int M, int N, int lda, float* hCPU, float* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float work;
    int   incx  = 1;
    float alpha = -1.0f;
    int   size  = lda * N;

    float cpu_norm = slange_(&norm_type, &M, &N, hCPU, &lda, &work);
    saxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = slange_(&norm_type, &M, &N, hGPU, &lda, &work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_general<double>(char norm_type, int M, int N, int lda, double* hCPU, double* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    size  = lda * N;

    double cpu_norm = dlange_(&norm_type, &M, &N, hCPU, &lda, work);
    daxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = dlange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
double norm_check_general<hipblasComplex>(
    char norm_type, int M, int N, int lda, hipblasComplex* hCPU, hipblasComplex* hGPU)
{
    //norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float work[1];
    int   incx  = 1;
    float alpha = -1.0f;
    int   size  = lda * N;

    float cpu_norm = clange_(&norm_type, &M, &N, hCPU, &lda, work);
    caxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = clange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_general<hipblasDoubleComplex>(
    char norm_type, int M, int N, int lda, hipblasDoubleComplex* hCPU, hipblasDoubleComplex* hGPU)
{
    //norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    size  = lda * N;

    double cpu_norm = zlange_(&norm_type, &M, &N, hCPU, &lda, work);
    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = zlange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

/* ============================Norm Check for Symmetric Matrix: float/double/complex template
 * speciliazation ======================================= */

/*! \brief compare the norm error of two hermitian/symmetric matrices hCPU & hGPU */

template <>
double
    norm_check_symmetric<float>(char norm_type, char uplo, int N, int lda, float* hCPU, float* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    float work[1];
    int   incx  = 1;
    float alpha = -1.0f;
    int   size  = lda * N;

    float cpu_norm = slansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    saxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    float error = slansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return (double)error;
}

template <>
double norm_check_symmetric<double>(
    char norm_type, char uplo, int N, int lda, double* hCPU, double* hGPU)
{
    // norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    double work[1];
    int    incx  = 1;
    double alpha = -1.0;
    int    size  = lda * N;

    double cpu_norm = dlansy_(&norm_type, &uplo, &N, hCPU, &lda, work);
    daxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = dlansy_(&norm_type, &uplo, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

// template<>
// double norm_check_symmetric<hipblasComplex>(char norm_type, char uplo, int N, int lda, hipblasComplex
// *hCPU, hipblasComplex *hGPU)
//{
////norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
//
//    float work[1];
//    int incx = 1;
//    float alpha = -1.0f;
//    int size = lda * N;
//
//    float cpu_norm = clanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
//    caxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);
//
//     float error = clanhe_(&norm_type, &uplo, &N, hGPU, &lda, work)/cpu_norm;
//
//    return (double)error;
//}
//
// template<>
// double norm_check_symmetric<hipblasDoubleComplex>(char norm_type, char uplo, int N, int lda,
// hipblasDoubleComplex *hCPU, hipblasDoubleComplex *hGPU)
//{
////norm type can be M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly
//
//    double work[1];
//    int incx = 1;
//    double alpha = -1.0;
//    int size = lda * N;
//
//    double cpu_norm = zlanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
//    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);
//
//     double error = zlanhe_(&norm_type, &uplo, &N, hGPU, &lda, work)/cpu_norm;
//
//    return error;
//}
