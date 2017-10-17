/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <sys/time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include "hipblas.hpp"
#include "utility.h"
#include "norm.h"
#include "unit.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template<typename T>
hipblasStatus_t testing_geam(Arguments argus)
{
    int M = argus.M;
    int N = argus.N;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    T h_alpha = argus.alpha;
    T h_beta = argus.beta;

    T *dA, *dB, *dC, *d_alpha, *d_beta;

    int  A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    int  inc1_A, inc2_A, inc1_B, inc2_B;

    T hipblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    hipblasCreate(&handle);

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M; A_col = N;
        inc1_A = 1; inc2_A = lda;
    }
    else
    {
        A_row = N; A_col = M;
        inc1_A = lda; inc2_A = 1;
    }
    if(transB == HIPBLAS_OP_N)
    {
        B_row = M; B_col = N;
        inc1_B = 1; inc2_B = ldb;
    }
    else
    {
        B_row = N; B_col = M;
        inc1_B = ldb; inc2_B = 1;
    }

    A_size = lda * A_col; B_size = ldb * B_col; C_size = ldc * N;

    //check here to prevent undefined memory allocation error
    if( M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M )
    {
        return HIPBLAS_STATUS_INVALID_VALUE; 
    }

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, C_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_alpha, sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_beta, sizeof(T)));

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC1(C_size);
    vector<T> hC2(C_size);
    vector<T> hC_copy(C_size);

    //Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, A_row, A_col, lda);
    hipblas_init<T>(hB, B_row, B_col, ldb);
    hipblas_init<T>(hC1, M, N, ldc);

    hC2 = hC1;
    hC_copy = hC1;

    //copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC1.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    // &h_alpha and &h_beta are host pointers
    status = hipblasGeam<T>(handle, transA, transB,
                M, N,
                &h_alpha, dA, lda,
                &h_beta, dB, ldb,
                dC, ldc);

    CHECK_HIP_ERROR(hipMemcpy(hC1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));


    // d_alpha and d_beta are device pointers
    CHECK_HIP_ERROR(hipMemcpy(dC, hC2.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    status = hipblasGeam<T>(handle, transA, transB,
                M, N,
                d_alpha, dA, lda,
                d_beta, dB, ldb,
                dC, ldc);

    //copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC2.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));


     /* =====================================================================
             CPU BLAS
     =================================================================== */
    if(status != HIPBLAS_STATUS_INVALID_VALUE) //only valid size compare with cblas
    {
        // reference calculation
        for (int i1 = 0; i1 < M; i1++)
        {
            for (int i2 = 0; i2 < N; i2++)
            {
                hC_copy[i1 + i2*ldc] = h_alpha * hA[i1*inc1_A + i2*inc2_A] + h_beta * hB[i1*inc1_B + i2*inc2_B];
            }
        }
    }

    #ifndef NDEBUG
    print_matrix(hC_copy, hC1, min(M,3), min(N,3), ldc);
    #endif

    //enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(argus.unit_check)
    {
        unit_check_general<T>(M, N, ldc, hC_copy.data(), hC1.data());
        unit_check_general<T>(M, N, ldc, hC_copy.data(), hC2.data());
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    hipblasDestroy(handle);
    return status;
}
