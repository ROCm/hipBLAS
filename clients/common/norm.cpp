/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "norm.h"
#include "cblas.h"
#include "hipblas/hipblas.h"
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
void caxpy_(
    int* n, hipblasComplex* alpha, hipblasComplex* x, int* incx, hipblasComplex* y, int* incy);
void zaxpy_(int*                  n,
            hipblasDoubleComplex* alpha,
            hipblasDoubleComplex* x,
            int*                  incx,
            hipblasDoubleComplex* y,
            int*                  incy);

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

    float          work[1];
    int            incx  = 1;
    hipblasComplex alpha = -1.0f;
    int            size  = lda * N;

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

    double               work[1];
    int                  incx  = 1;
    hipblasDoubleComplex alpha = -1.0;
    int                  size  = lda * N;

    double cpu_norm = zlange_(&norm_type, &M, &N, hCPU, &lda, work);
    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);

    double error = zlange_(&norm_type, &M, &N, hGPU, &lda, work) / cpu_norm;

    return error;
}

template <>
double norm_check_general<hipblasHalf>(
    char norm_type, int M, int N, int lda, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<double> hCPU_double(N * lda);
    host_vector<double> hGPU_double(N * lda);

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hCPU_double[i + j * lda] = hCPU[i + j * lda];
            hGPU_double[i + j * lda] = hGPU[i + j * lda];
        }
    }

    return norm_check_general<double>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

template <>
double norm_check_general<hipblasBfloat16>(
    char norm_type, int M, int N, int lda, hipblasBfloat16* hCPU, hipblasBfloat16* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<float> hCPU_double(N * lda);
    host_vector<float> hGPU_double(N * lda);

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hCPU_double[i + j * lda] = bfloat16_to_float(hCPU[i + j * lda]);
            hGPU_double[i + j * lda] = bfloat16_to_float(hGPU[i + j * lda]);
        }
    }

    return norm_check_general<float>(norm_type, M, N, lda, hCPU_double, hGPU_double);
}

template <>
double
    norm_check_general<int32_t>(char norm_type, int M, int N, int lda, int32_t* hCPU, int32_t* hGPU)
{
    // norm type can be 'M', 'I', 'F', 'l': 'F' (Frobenius norm) is used mostly

    host_vector<float> hCPU_float(N * lda);
    host_vector<float> hGPU_float(N * lda);

    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hCPU_float[i + j * lda] = (hCPU[i + j * lda]);
            hGPU_float[i + j * lda] = (hGPU[i + j * lda]);
        }
    }

    return norm_check_general<float>(norm_type, M, N, lda, hCPU_float, hGPU_float);
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
//    hipblasComplex alpha = -1.0f;
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
//    hipblasDoubleComplex alpha = -1.0;
//    int size = lda * N;
//
//    double cpu_norm = zlanhe_(&norm_type, &uplo, &N, hCPU, &lda, work);
//    zaxpy_(&size, &alpha, hCPU, &incx, hGPU, &incx);
//
//     double error = zlanhe_(&norm_type, &uplo, &N, hGPU, &lda, work)/cpu_norm;
//
//    return error;
//}
