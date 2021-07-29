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
hipblasStatus_t testing_hemm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHemmStridedBatchedFn
        = FORTRAN ? hipblasHemmStridedBatched<T, true> : hipblasHemmStridedBatched<T, false>;

    int    M            = argus.M;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    int    ldc          = argus.ldc;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);
    hipblasFillMode_t uplo = char2hipblas_fill(argus.uplo_option);

    int           K        = (side == HIPBLAS_SIDE_LEFT ? M : N);
    hipblasStride stride_A = size_t(lda) * K * stride_scale;
    hipblasStride stride_B = size_t(ldb) * N * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;

    size_t A_size = size_t(stride_A) * batch_count;
    size_t B_size = size_t(stride_B) * batch_count;
    size_t C_size = size_t(stride_C) * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda, stride_A, batch_count);
    hipblas_init<T>(hB, M, N, ldb, stride_B, batch_count);
    hipblas_init<T>(hC_host, M, N, ldc, stride_C, batch_count);
    hC_gold   = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasHemmStridedBatchedFn(handle,
                                                    side,
                                                    uplo,
                                                    M,
                                                    N,
                                                    &h_alpha,
                                                    dA,
                                                    lda,
                                                    stride_A,
                                                    dB,
                                                    ldb,
                                                    stride_B,
                                                    &h_beta,
                                                    dC,
                                                    ldc,
                                                    stride_C,
                                                    batch_count));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasHemmStridedBatchedFn(handle,
                                                    side,
                                                    uplo,
                                                    M,
                                                    N,
                                                    d_alpha,
                                                    dA,
                                                    lda,
                                                    stride_A,
                                                    dB,
                                                    ldb,
                                                    stride_B,
                                                    d_beta,
                                                    dC,
                                                    ldc,
                                                    stride_C,
                                                    batch_count));

    CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_hemm<T>(side,
                          uplo,
                          M,
                          N,
                          h_alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hB.data() + b * stride_B,
                          ldb,
                          h_beta,
                          hC_gold.data() + b * stride_C,
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_gold, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_gold, hC_device);
        }

        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_gold, hC_device, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasHemmStridedBatchedFn(handle,
                                                            side,
                                                            uplo,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_ldb,
                      e_stride_b,
                      e_beta,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         hemm_gflop_count<T>(M, N, K),
                         hemm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
