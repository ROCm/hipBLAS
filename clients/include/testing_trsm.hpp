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
hipblasStatus_t testing_trsm(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTrsmFn = FORTRAN ? hipblasTrsm<T, true> : hipblasTrsm<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T    alpha       = argus.alpha;

    hipblasSideMode_t  side   = char2hipblas_side(char_side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(char_diag);

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hB_copy(B_size);
    host_vector<T> hX(B_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, K, lda);
    // pad untouched area into zero
    for(int i = K; i < lda; i++)
    {
        for(int j = 0; j < K; j++)
        {
            hA[i + j * lda] = 0.0;
        }
    }
    // proprocess the matrix to avoid ill-conditioned matrix
    vector<int> ipiv(K);
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
    hipblas_init<T>(hB, M, N, ldb);
    // pad untouched area into zero
    for(int i = M; i < ldb; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hB[i + j * ldb] = 0.0;
        }
    }
    hX = hB; // original solution hX

    // Calculate hB = hA*hX;
    cblas_trmm<T>(
        side, uplo, transA, diag, M, N, T(1.0) / alpha, (const T*)hA.data(), lda, hB.data(), ldb);

    hB_copy = hB;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasTrsmFn(handle, side, uplo, transA, diag, M, N, &alpha, dA, lda, dB, ldb);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB.data(), dB, sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_trsm<T>(
            side, uplo, transA, diag, M, N, alpha, (const T*)hA.data(), lda, hB_copy.data(), ldb);

        //      print_matrix(hB_copy, hB, min(M, 3), min(N,3), ldb);

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        double error = norm_check_general<T>('F', M, N, ldb, hB_copy.data(), hB.data());
        unit_check_error(error, tolerance);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
