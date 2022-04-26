/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

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

    int           dim_AP   = N * (N + 1) / 2;
    int           abs_incx = incx < 0 ? -incx : incx;
    hipblasStride strideA  = N * N; // only for test setup
    hipblasStride strideAP = dim_AP * stride_scale;
    hipblasStride stridex  = abs_incx * N * stride_scale;
    size_t        size_A   = strideA * batch_count;
    size_t        size_AP  = strideAP * batch_count;
    size_t        size_x   = stridex * batch_count;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, nullptr, strideAP, nullptr, incx, stridex, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
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

    double gpu_time_used, hipblas_error, cumulative_hipblas_error = 0;
    // Initial Data on CPU
    hipblas_init_matrix(
        hA, argus, N, N, N, strideA, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, argus, N, abs_incx, stridex, batch_count, hipblas_client_never_set_nan, false, true);
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
                t += std::abs(AATb[i + j * N]);
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
    CHECK_HIP_ERROR(hipMemcpy(dAP, hAP.data(), sizeof(T) * size_AP, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
            handle, uplo, transA, diag, N, dAP, strideAP, dx_or_b, incx, stridex, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(
                N, abs_incx, hx.data() + b * stridex, hx_or_b_1.data() + b * stridex));
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
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTpsvStridedBatchedFn(
                handle, uplo, transA, diag, N, dAP, strideAP, dx_or_b, incx, stridex, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_N,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         tpsv_gflop_count<T>(N),
                         tpsv_gbyte_count<T>(N),
                         cumulative_hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
