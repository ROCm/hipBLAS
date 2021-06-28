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
hipblasStatus_t testing_hemm(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasHemmFn = FORTRAN ? hipblasHemm<T, true> : hipblasHemm<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    char char_side = argus.side_option;
    char char_uplo = argus.uplo_option;
    T    h_alpha   = argus.get_alpha<T>();
    T    h_beta    = argus.get_beta<T>();

    hipblasSideMode_t side = char2hipblas_side(char_side);
    hipblasFillMode_t uplo = char2hipblas_fill(char_uplo);

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || ldc < M || ldb < M || lda < K)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;
    size_t C_size = size_t(ldc) * N;

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

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hB, M, N, ldb);
    hipblas_init<T>(hC_host, M, N, ldc);
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
    CHECK_HIPBLAS_ERROR(
        hipblasHemmFn(handle, side, uplo, M, N, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(
        hipblasHemmFn(handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

    CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_hemm<T>(
            side, uplo, M, N, h_alpha, hA.data(), lda, hB.data(), ldb, h_beta, hC_gold.data(), ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_host);
            unit_check_general<T>(M, N, ldc, hC_gold, hC_device);
        }

        if(argus.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_host);
            hipblas_error_device = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_device);
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

            CHECK_HIPBLAS_ERROR(hipblasHemmFn(
                handle, side, uplo, M, N, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_beta,
                      e_ldc>{}
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
