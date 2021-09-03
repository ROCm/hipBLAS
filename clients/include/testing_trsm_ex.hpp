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

#define TRSM_BLOCK 128

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsm_ex(const Arguments& argus)
{
    bool FORTRAN         = argus.fortran;
    auto hipblasTrsmExFn = FORTRAN ? hipblasTrsmExFortran : hipblasTrsmEx;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T    h_alpha     = argus.get_alpha<T>();

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int    K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB_host(B_size);
    host_vector<T> hB_device(B_size);
    host_vector<T> hB_cpu(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dinvA(TRSM_BLOCK * K);
    device_vector<T> d_alpha(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial hA on CPU
    srand(1);
    hipblas_init(hA, K, K, lda);
    // pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = 0.0;
        }
    }
    // proprocess the matrix to avoid ill-conditioned matrix
    host_vector<int> ipiv(K);
    cblas_getrf(K, K, hA.data(), lda, ipiv.data());
    for(int i = 0; i < K; i++)
    {
        for(int j = i; j < K; j++)
        {
            hA[i + j * lda] = hA[j + i * lda];
            if(diag == HIPBLAS_DIAG_UNIT)
            {
                if(i == j)
                    hA[i + j * lda] = 1.0;
            }
        }
    }

    // Initial hB, hX on CPU
    hipblas_init<T>(hB_host, M, N, ldb);
    // pad untouched area into zero
    for(int i = M; i < ldb; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hB_host[i + j * ldb] = 0.0;
        }
    }

    // Calculate hB = hA*hX;
    cblas_trmm<T>(side,
                  uplo,
                  transA,
                  diag,
                  M,
                  N,
                  T(1.0) / h_alpha,
                  (const T*)hA.data(),
                  lda,
                  hB_host.data(),
                  ldb);

    hB_cpu = hB_device = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB_host, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipblasStride stride_A    = TRSM_BLOCK * size_t(lda) + TRSM_BLOCK;
    hipblasStride stride_invA = TRSM_BLOCK * TRSM_BLOCK;
    int           blocks      = K / TRSM_BLOCK;

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    // Calculate invA
    if(blocks > 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                          uplo,
                                                          diag,
                                                          TRSM_BLOCK,
                                                          dA,
                                                          lda,
                                                          stride_A,
                                                          dinvA,
                                                          TRSM_BLOCK,
                                                          stride_invA,
                                                          blocks));
    }

    if(K % TRSM_BLOCK != 0 || blocks == 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                          uplo,
                                                          diag,
                                                          K - TRSM_BLOCK * blocks,
                                                          dA + stride_A * blocks,
                                                          lda,
                                                          stride_A,
                                                          dinvA + stride_invA * blocks,
                                                          TRSM_BLOCK,
                                                          stride_invA,
                                                          1));
    }

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                        side,
                                        uplo,
                                        transA,
                                        diag,
                                        M,
                                        N,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        dB,
                                        ldb,
                                        dinvA,
                                        TRSM_BLOCK * K,
                                        argus.compute_type));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB_host, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB_device, sizeof(T) * B_size, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                        side,
                                        uplo,
                                        transA,
                                        diag,
                                        M,
                                        N,
                                        d_alpha,
                                        dA,
                                        lda,
                                        dB,
                                        ldb,
                                        dinvA,
                                        TRSM_BLOCK * K,
                                        argus.compute_type));

    CHECK_HIP_ERROR(hipMemcpy(hB_device, dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_trsm<T>(
            side, uplo, transA, diag, M, N, h_alpha, (const T*)hA.data(), lda, hB_cpu.data(), ldb);

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host   = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_host);
        hipblas_error_device = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_device);

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
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                                side,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                N,
                                                d_alpha,
                                                dA,
                                                lda,
                                                dB,
                                                ldb,
                                                dinvA,
                                                TRSM_BLOCK * K,
                                                argus.compute_type));
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
                      e_ldb>{}
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
