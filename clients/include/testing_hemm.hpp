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
    T    alpha     = argus.get_alpha<T>();
    T    beta      = argus.get_beta<T>();

    hipblasSideMode_t side = char2hipblas_side(char_side);
    hipblasFillMode_t uplo = char2hipblas_fill(char_uplo);

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || ldc < M || ldb < M || lda < K)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    int A_size = size_t(lda) * K;
    int B_size = size_t(ldb) * N;
    int C_size = size_t(ldc) * N;

    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_1(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hB, M, N, ldb);
    hipblas_init<T>(hC_1, M, N, ldc);
    hC_gold = hC_1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_1.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status = hipblasHemmFn(handle, side, uplo, M, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_hemm<T>(
            side, uplo, M, N, alpha, hA.data(), lda, hB.data(), ldb, beta, hC_gold.data(), ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold.data(), hC_1.data());
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
