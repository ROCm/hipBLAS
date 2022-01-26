/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trmm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmmStridedBatchedFn
        = FORTRAN ? hipblasTrmmStridedBatched<T, true> : hipblasTrmmStridedBatched<T, false>;

    int    M            = argus.M;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int           K        = (side == HIPBLAS_SIDE_LEFT ? M : N);
    hipblasStride stride_A = size_t(lda) * K * stride_scale;
    hipblasStride stride_B = size_t(ldb) * N * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t B_size = stride_B * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB_host(B_size);
    host_vector<T> hB_device(B_size);
    host_vector<T> hB_gold(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> d_alpha(1);

    T h_alpha = argus.get_alpha<T>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, K, lda, stride_A, batch_count);
    hipblas_init<T>(hB_host, M, N, ldb, stride_B, batch_count);
    hB_device = hB_host;
    hB_gold   = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB_host, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hB_host, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_device, sizeof(T) * B_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hB_device, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          h_alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hB_gold.data() + b * stride_B,
                          ldb);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldb, stride_B, hB_gold, hB_host);
            unit_check_general<T>(M, N, batch_count, ldb, stride_B, hB_gold, hB_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldb, stride_B, hB_gold, hB_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldb, stride_B, hB_gold, hB_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasTrmmStridedBatchedFn(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_lda,
                      e_ldb,
                      e_stride_b,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, K),
                         trmm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
