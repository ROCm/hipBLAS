/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
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
hipblasStatus_t testing_tpsv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTpsvBatchedFn
        = FORTRAN ? hipblasTpsvBatched<T, true> : hipblasTpsvBatched<T, false>;

    int                N           = argus.N;
    int                incx        = argus.incx;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = N * N;
    int size_AP  = N * (N + 1) / 2;
    int size_x   = abs_incx * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hAP[batch_count];
    host_vector<T> AAT[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_or_b_1[batch_count];
    host_vector<T> hx_or_b_2[batch_count];
    host_vector<T> cpu_x_or_b[batch_count];

    device_batch_vector<T> bAP(batch_count, size_AP);
    device_batch_vector<T> bx_or_b(batch_count, size_x);

    device_vector<T*, 0, T> dAP(batch_count);
    device_vector<T*, 0, T> dx_or_b(batch_count);

    int last = batch_count - 1;
    if(!dAP || !dx_or_b || (!bAP[last] && size_AP) || (!bx_or_b[last] && size_x))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblas_error, cumulative_hipblas_error = 0;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]         = host_vector<T>(size_A);
        hAP[b]        = host_vector<T>(size_AP);
        AAT[b]        = host_vector<T>(size_A);
        hb[b]         = host_vector<T>(size_x);
        hx[b]         = host_vector<T>(size_x);
        hx_or_b_1[b]  = host_vector<T>(size_x);
        hx_or_b_2[b]  = host_vector<T>(size_x);
        cpu_x_or_b[b] = host_vector<T>(size_x);

        srand(1);
        hipblas_init<T>(hA[b], N, N, N);

        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(HIPBLAS_OP_N,
                      HIPBLAS_OP_T,
                      N,
                      N,
                      N,
                      (T)1.0,
                      (T*)hA[b],
                      N,
                      (T*)hA[b],
                      N,
                      (T)0.0,
                      (T*)AAT[b],
                      N);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < N; i++)
        {
            T t = 0.0;
            for(int j = 0; j < N; j++)
            {
                hA[b][i + j * N] = AAT[b][i + j * N];
                t += abs(AAT[b][i + j * N]);
            }
            hA[b][i + i * N] = t;
        }

        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, N, hA[b], N);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < N; i++)
                {
                    T diag = hA[b][i + i * N];
                    for(int j = 0; j <= i; j++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
            else
                for(int j = 0; j < N; j++)
                {
                    T diag = hA[b][j + j * N];
                    for(int i = 0; i <= j; i++)
                        hA[b][i + j * N] = hA[b][i + j * N] / diag;
                }
        }

        hipblas_init<T>(hx[b], 1, N, abs_incx);
        hb[b] = hx[b];

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, N, hA[b], N, hb[b], incx);
        cpu_x_or_b[b] = hb[b]; // cpuXorB <- B
        hx_or_b_1[b]  = hb[b];
        hx_or_b_2[b]  = hb[b];

        regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], (T*)hAP[b], N);

        // copy data from CPU to device
        hipMemcpy(bAP[b], hAP[b], sizeof(T) * size_AP, hipMemcpyHostToDevice);
        hipMemcpy(bx_or_b[b], hx_or_b_1[b], sizeof(T) * size_x, hipMemcpyHostToDevice);
    }
    hipMemcpy(dAP, bAP, sizeof(T*) * batch_count, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, bx_or_b, sizeof(T*) * batch_count, hipMemcpyHostToDevice);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        status
            = hipblasTpsvBatchedFn(handle, uplo, transA, diag, N, dAP, dx_or_b, incx, batch_count);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        // copy output from device to CPU
        for(int b = 0; b < batch_count; b++)
        {
            hipMemcpy(hx_or_b_1[b], bx_or_b[b], sizeof(T) * size_x, hipMemcpyDeviceToHost);
        }

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error
                = std::abs(vector_norm_1<T>(N, abs_incx, hx[b].data(), hx_or_b_1[b].data()));
            if(argus.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * N;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }
    if(argus.timing)
    {
        hipStream_t stream;
        status = hipblasGetStream(handle, &stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            status = hipblasTpsvBatchedFn(
                handle, uplo, transA, diag, N, dAP, dx_or_b, incx, batch_count);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_N, e_incx, e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         tpsv_gflop_count<T>(N),
                         tpsv_gbyte_count<T>(N),
                         cumulative_hipblas_error);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
