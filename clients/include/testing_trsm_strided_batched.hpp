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
hipblasStatus_t testing_trsm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrsmStridedBatchedFn
        = FORTRAN ? hipblasTrsmStridedBatched<T, true> : hipblasTrsmStridedBatched<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char   char_side    = argus.side_option;
    char   char_uplo    = argus.uplo_option;
    char   char_transA  = argus.transA_option;
    char   char_diag    = argus.diag_option;
    T      h_alpha      = argus.get_alpha<T>();
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasStride strideA = size_t(lda) * K * stride_scale;
    hipblasStride strideB = size_t(ldb) * N * stride_scale;
    size_t        A_size  = strideA * batch_count;
    size_t        B_size  = strideB * batch_count;

    // check here to prevent undefined memory allocation error
    // TODO: Workaround for cuda tests, not actually testing return values
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB_host(B_size);
    host_vector<T> hB_device(B_size);
    host_vector<T> hB_gold(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> d_alpha(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial hA on CPU
    hipblas_init_matrix(
        hA, argus, K, K, lda, strideA, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_matrix(
        hB_host, argus, M, N, ldb, strideB, batch_count, hipblas_client_never_set_nan);

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;
        T* hBb = hB_host.data() + b * strideB;

        // pad ountouched area into zero
        for(int i = K; i < lda; i++)
        {
            for(int j = 0; j < K; j++)
            {
                hAb[i + j * lda] = 0.0;
            }
        }

        // proprocess the matrix to avoid ill-conditioned matrix
        std::vector<int> ipiv(K);
        cblas_getrf(K, K, hAb, lda, ipiv.data());
        for(int i = 0; i < K; i++)
        {
            for(int j = i; j < K; j++)
            {
                hAb[i + j * lda] = hAb[j + i * lda];
                if(diag == HIPBLAS_DIAG_UNIT)
                {
                    if(i == j)
                        hAb[i + j * lda] = 1.0;
                }
            }
        }

        // pad untouched area into zero
        for(int i = M; i < ldb; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hBb[i + j * ldb] = 0.0;
            }
        }

        // Calculate hB = hA*hX;
        cblas_trmm<T>(
            side, uplo, transA, diag, M, N, T(1.0) / h_alpha, (const T*)hAb, lda, hBb, ldb);
    }
    hB_gold   = hB_host; // original solutions hX
    hB_device = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB_host, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrsmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        dB,
                                                        ldb,
                                                        strideB,
                                                        batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hB_host, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_device, sizeof(T) * B_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrsmStridedBatchedFn(handle,
                                                        side,
                                                        uplo,
                                                        transA,
                                                        diag,
                                                        M,
                                                        N,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        strideA,
                                                        dB,
                                                        ldb,
                                                        strideB,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hB_device, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_trsm<T>(side,
                          uplo,
                          transA,
                          diag,
                          M,
                          N,
                          h_alpha,
                          (const T*)hA.data() + b * strideA,
                          lda,
                          hB_gold.data() + b * strideB,
                          ldb);
        }

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host
            = norm_check_general<T>('F', M, N, ldb, strideB, hB_gold, hB_host, batch_count);
        hipblas_error_device
            = norm_check_general<T>('F', M, N, ldb, strideB, hB_gold, hB_device, batch_count);
        if(argus.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
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
            {
                gpu_time_used = get_time_us_sync(stream);
            }

            CHECK_HIPBLAS_ERROR(hipblasTrsmStridedBatchedFn(handle,
                                                            side,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dB,
                                                            ldb,
                                                            strideB,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_ldb,
                      e_stride_b,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trsm_gflop_count<T>(M, N, K),
                         trsm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
