/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_tpsv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTpsvStridedBatchedFn
        = FORTRAN ? hipblasTpsvStridedBatched<T, true> : hipblasTpsvStridedBatched<T, false>;

    int                N            = argus.N;
    int                incx         = argus.incx;
    char               char_uplo    = argus.uplo_option;
    char               char_diag    = argus.diag_option;
    char               char_transA  = argus.transA_option;
    hipblasFillMode_t  uplo         = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA       = char2hipblas_operation(char_transA);
    double             stride_scale = argus.stride_scale;
    int                batch_count  = argus.batch_count;

    int dim_AP   = N * (N + 1) / 2;
    int abs_incx = incx < 0 ? -incx : incx;
    int strideA  = N * N; // only for test setup
    int strideAP = dim_AP * stride_scale;
    int stridex  = abs_incx * N * stride_scale;
    int size_A   = strideA * batch_count;
    int size_AP  = strideAP * batch_count;
    int size_x   = stridex * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hAP(size_AP);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    device_vector<T> dAP(size_AP);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, N, N, strideA, batch_count);
    hipblas_init<T>(hx, 1, N, abs_incx, stridex, batch_count);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb  = hA.data() + b * strideA;
        T* hAPb = hAP.data() + b * strideAP;
        T* AATb = AAT.data() + b * strideA;
        T* hbb  = hb.data() + b * stridex;
        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(HIPBLAS_OP_N, HIPBLAS_OP_T, N, N, N, (T)1.0, hAb, N, hAb, N, (T)0.0, AATb, N);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < N; i++)
        {
            T t = 0.0;
            for(int j = 0; j < N; j++)
            {
                hAb[i + j * N] = AATb[i + j * N];
                t += abs(AATb[i + j * N]);
            }
            hAb[i + i * N] = t;
        }
        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, N, hAb, N);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < N; i++)
                {
                    T diag = hAb[i + i * N];
                    for(int j = 0; j <= i; j++)
                        hAb[i + j * N] = hAb[i + j * N] / diag;
                }
            else
                for(int j = 0; j < N; j++)
                {
                    T diag = hAb[j + j * N];
                    for(int i = 0; i <= j; i++)
                        hAb[i + j * N] = hA[b + j * N] / diag;
                }
        }

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hAb, N, hbb, incx);

        regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hAb, (T*)hAPb, N);
    }

    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // copy data from CPU to device
    hipMemcpy(dAP, hAP.data(), sizeof(T) * size_AP, hipMemcpyHostToDevice);
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
        status = hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, dAP, strideAP, dx_or_b, incx, stridex, batch_count);

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
            real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
            double    tolerance = eps * 40 * N;

            T*     hxb        = hx + b * stridex;
            T*     hx_or_b_1b = hx_or_b_1 + b * stridex;
            double error = 0.0, max_err_scal = 0.0, max_err = 0.0;
            for(int i = 0; i < N; i++)
            {
                T diff = (hxb[i * abs_incx] - hx_or_b_1b[i * abs_incx]);
                if(diff != T(0))
                {
                    max_err += abs(diff);
                }
                max_err_scal += abs(hx_or_b_1b[i * abs_incx]);
            }
            error = max_err / max_err_scal;

            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
