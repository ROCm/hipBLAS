/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsv(Arguments argus)
{
    constexpr T eps      = std::numeric_limits<T>::epsilon();
    constexpr T eps_mult = 40; // arbitrary

    int                M           = argus.M;
    int                incx        = argus.incx;
    int                lda         = argus.lda;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = lda * M;
    int size_x   = abs_incx * M;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(lda < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incx == 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(size_A);
    vector<T> AAT(size_A);
    vector<T> hb(size_x);
    vector<T> hx(size_x);
    vector<T> hx_or_b_1(size_x);
    vector<T> hx_or_b_2(size_x);
    vector<T> cpu_x_or_b(size_x);

    T *dA, *dx_or_b;
    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, size_A * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx_or_b, size_x * sizeof(T)));

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda);

    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T>(HIPBLAS_OP_N,
                  HIPBLAS_OP_T,
                  M,
                  M,
                  M,
                  1.0,
                  hA.data(),
                  lda,
                  hA.data(),
                  lda,
                  0.0,
                  AAT.data(),
                  lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < M; i++)
    {
        T t = 0.0;
        for(int j = 0; j < M; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += AAT[i + j * lda] > 0 ? AAT[i + j * lda] : -AAT[i + j * lda];
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, M, hA.data(), lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
            for(int i = 0; i < M; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
        else
            for(int j = 0; j < M; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
    }

    hipblas_init<T>(hx, 1, M, abs_incx);
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, M, hA.data(), lda, hb.data(), incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasTrsv<T>(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            CHECK_HIP_ERROR(hipFree(dA));
            CHECK_HIP_ERROR(hipFree(dx_or_b));
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        double error = 0.0;
        if(argus.unit_check)
        {
            for(int i = 0; i < M; i++)
            {
                if(hx[i * abs_incx] != 0)
                    error += std::abs((hx[i * abs_incx] - hx_or_b_1[i * abs_incx])
                                      / hx[i * abs_incx]);
                else
                    error += std::abs(hx_or_b_1[i * abs_incx]);
            }
        }
        unit_check_trsv(error, M, eps_mult, eps);
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx_or_b));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
