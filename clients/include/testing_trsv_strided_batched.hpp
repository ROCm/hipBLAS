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
hipblasStatus_t testing_trsv_strided_batched(Arguments argus)
{
    constexpr T eps      = std::numeric_limits<T>::epsilon();
    constexpr T eps_mult = 40; // arbitrary

    int                M            = argus.M;
    int                incx         = argus.incx;
    int                lda          = argus.lda;
    char               char_uplo    = argus.uplo_option;
    char               char_diag    = argus.diag_option;
    char               char_transA  = argus.transA_option;
    hipblasFillMode_t  uplo         = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA       = char2hipblas_operation(char_transA);
    double             stride_scale = argus.stride_scale;
    int                batch_count  = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int strideA  = lda * M * stride_scale;
    int stridex  = abs_incx * M * stride_scale;
    int size_A   = strideA * batch_count;
    int size_x   = stridex * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || lda < M || !incx || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count || !M || !lda)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda, strideA, batch_count);
    hipblas_init<T>(hx, 1, M, abs_incx, stridex, batch_count);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb  = hA.data() + b * strideA;
        T* AATb = AAT.data() + b * strideA;
        T* hbb  = hb.data() + b * stridex;
        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(HIPBLAS_OP_N, HIPBLAS_OP_T, M, M, M, 1.0, hAb, lda, hAb, lda, 0.0, AATb, lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < M; i++)
        {
            T t = 0.0;
            for(int j = 0; j < M; j++)
            {
                hAb[i + j * lda] = AATb[i + j * lda];
                t += AATb[i + j * lda] > 0 ? AATb[i + j * lda] : -AATb[i + j * lda];
            }
            hAb[i + i * lda] = t;
        }
        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, M, hAb, lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < M; i++)
                {
                    T diag = hAb[i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hAb[i + j * lda] = hAb[i + j * lda] / diag;
                }
            else
                for(int j = 0; j < M; j++)
                {
                    T diag = hAb[j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hAb[i + j * lda] = hA[b + j * lda] / diag;
                }
        }

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, M, hAb, lda, hbb, incx);
    }

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
        status = hipblasTrsvStridedBatched<T>(
            handle, uplo, transA, diag, M, dA, lda, strideA, dx_or_b, incx, stridex, batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            T*     hxb        = hx + b * stridex;
            T*     hx_or_b_1b = hx_or_b_1 + b * stridex;
            double error      = 0.0;
            for(int i = 0; i < M; i++)
            {
                if(hx[i * abs_incx] != 0)
                    error += std::abs((hxb[i * abs_incx] - hx_or_b_1b[i * abs_incx])
                                      / hxb[i * abs_incx]);
                else
                    error += std::abs(hx_or_b_1b[i * abs_incx]);
            }
            unit_check_trsv(error, M, eps_mult, eps);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
