/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <limits>
#include <memory>
#include <iostream>
#include "hipblas.h"
#include "utility.h"

using namespace std;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error) \
if (error != hipSuccess) { \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
}
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error) \
if (error != HIPBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "rocBLAS error: "); \
    if(error == HIPBLAS_STATUS_NOT_INITIALIZED)fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED"); \
    if(error == HIPBLAS_STATUS_ALLOC_FAILED)fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED"); \
    if(error == HIPBLAS_STATUS_INVALID_VALUE)fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE"); \
    if(error == HIPBLAS_STATUS_MAPPING_ERROR)fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR"); \
    if(error == HIPBLAS_STATUS_EXECUTION_FAILED)fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
    if(error == HIPBLAS_STATUS_INTERNAL_ERROR)fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR"); \
    if(error == HIPBLAS_STATUS_NOT_SUPPORTED)fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED"); \
    fprintf(stderr, "\n"); \
    exit(EXIT_FAILURE); \
}
#endif

#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

void mat_mat_mult(hipblasHalf alpha, hipblasHalf beta, int M, int N, int K, hipblasHalf* A, int As1, int As2,
                           hipblasHalf* B, int Bs1, int Bs2, hipblasHalf* C, int Cs1, int Cs2) 
{

    float alpha_float = half_to_float(alpha);
    float beta_float  = half_to_float(beta);

    int sizeA = M * As1 > K * As2 ? M * As1 : K * As2;
    int sizeB = K * Bs1 > N * Bs2 ? K * Bs1 : N * Bs2;
    int sizeC = N * Cs2;

    std::unique_ptr<float[]> A_float(new float[sizeA]());
    std::unique_ptr<float[]> B_float(new float[sizeB]());
    std::unique_ptr<float[]> C_float(new float[sizeC]());

    for(int i = 0; i < sizeA; i++)
    {
        A_float[i] = half_to_float(A[i]);
    }
    for(int i = 0; i < sizeB; i++)
    {
        B_float[i] = half_to_float(B[i]);
    }
    for(int i = 0; i < sizeC; i++)
    {
        C_float[i] = half_to_float(C[i]);
    }

    for(int i1=0; i1<M; i1++) 
    {
        for(int i2=0; i2<N; i2++) 
        {
            float t = 0.0;
            for(int i3=0; i3<K; i3++)
            {
                t +=  A_float[i1 * As1 + i3 * As2] * B_float[i3 * Bs1 + i2 * Bs2];
            }
            C_float[i1*Cs1 +i2*Cs2] = beta * C_float[i1*Cs1+i2*Cs2] + alpha * t ;
        }
    }
    for(int i = 0; i < sizeC; i++)
    {
        C[i] = float_to_half(C_float[i]);
    }

}

int main()
{
    hipblasOperation_t transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_T;
    hipblasHalf alpha = float_to_half((float)1.1), beta = float_to_half((float)0.9);

    int m = DIM1, n = DIM2, k = DIM3;
    int lda, ldb, ldc, size_a, size_b, size_c;
    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    cout << "hgemm example" << endl;
    if (transa == HIPBLAS_OP_N)
    {
        lda = m;
        size_a = k * lda;
        a_stride_1 = 1; a_stride_2 = lda;
        cout << "N";
    }
    else
    {
        lda = k;
        size_a = m * lda;
        a_stride_1 = lda; a_stride_2 = 1;
        cout << "T";
    }
    if (transb == HIPBLAS_OP_N)
    {
        ldb = k;
        size_b = n * ldb;
        b_stride_1 = 1; b_stride_2 = ldb;
        cout << "N: ";
    }
    else
    {
        ldb = n;
        size_b = k * ldb;
        b_stride_1 = ldb; b_stride_2 = 1;
        cout << "T: ";
    }
    ldc = m;
    size_c = n * ldc;

    //Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    vector<hipblasHalf> ha(size_a);
    vector<hipblasHalf> hb(size_b);
    vector<hipblasHalf> hc(size_c);
    vector<hipblasHalf> hc_gold(size_c);

    //initial data on host
    srand(1);
    for( int i = 0; i < size_a; ++i ) { ha[i] = float_to_half((float)(rand() % 17)); }
    for( int i = 0; i < size_b; ++i ) { hb[i] = float_to_half((float)(rand() % 17)); }
    for( int i = 0; i < size_c; ++i ) { hc[i] = float_to_half((float)(rand() % 17)); }
    hc_gold = hc;

    //allocate memory on device
    hipblasHalf *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(hipblasHalf)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(hipblasHalf)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(hipblasHalf)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(hipblasHalf) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(hipblasHalf) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(hipblasHalf) * size_c, hipMemcpyHostToDevice));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));
    
    CHECK_HIPBLAS_ERROR(hipblasHgemm(handle, transa, transb, m, n, k, &alpha, 
                da, lda,
                db, ldb, &beta, 
                dc, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(hipblasHalf) * size_c, hipMemcpyDeviceToHost));

    cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda << ", " << ldb <<  ", " << ldc << endl;

    float max_relative_error = numeric_limits<float>::min();

    // calculate golden or correct result
    mat_mat_mult(alpha, beta, m, n, k, 
            ha.data(), a_stride_1, a_stride_2, 
            hb.data(), b_stride_1, b_stride_2, 
            hc_gold.data(), 1, ldc);

    for (int i = 0; i < size_c; i++)
    {
        float relative_error = (half_to_float(hc_gold[i]) - half_to_float(hc[i])) / half_to_float(hc_gold[i]);
        relative_error = relative_error > 0 ? relative_error : -relative_error;
        max_relative_error = relative_error < max_relative_error ? max_relative_error : relative_error;
    }
    float eps = numeric_limits<float>::epsilon();
    float tolerance = 10;
    if (max_relative_error != max_relative_error || max_relative_error > eps * tolerance)
    {
        cout << "FAIL: max_relative_error = " << max_relative_error << endl;
    }
    else
    {
        cout << "PASS: max_relative_error = " << max_relative_error << endl;
    }

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    return EXIT_SUCCESS;
}
