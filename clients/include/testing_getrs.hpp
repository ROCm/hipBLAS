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

template <typename T, typename U>
hipblasStatus_t testing_getrs(const Arguments& argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasGetrsFn = FORTRAN ? hipblasGetrs<T, true> : hipblasGetrs<T, false>;

    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    int A_size    = lda * N;
    int B_size    = ldb * 1;
    int Ipiv_size = N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA(A_size);
    host_vector<T>   hX(B_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB1(B_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    int              info;

    device_vector<T>   dA(A_size);
    device_vector<T>   dB(B_size);
    device_vector<int> dIpiv(Ipiv_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblas_init<T>(hA, N, N, lda);
    hipblas_init<T>(hX, N, 1, ldb);

    // scale A to avoid singularities
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(i == j)
                hA[i + j * lda] += 400;
            else
                hA[i + j * lda] -= 4;
        }
    }

    // Calculate hB = hA*hX;
    hipblasOperation_t op = HIPBLAS_OP_N;
    cblas_gemm<T>(op, op, N, 1, N, (T)1, hA.data(), lda, hX.data(), ldb, (T)0, hB.data(), ldb);

    // LU factorize hA on the CPU
    info = cblas_getrf<T>(N, N, hA.data(), lda, hIpiv.data());
    if(info != 0)
    {
        cerr << "LU decomposition failed" << endl;
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), B_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrsFn(handle, op, N, 1, dA, lda, dIpiv, dB, ldb, &info);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB1.data(), dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        cblas_getrs('N', N, 1, hA.data(), lda, hIpiv.data(), hB.data(), ldb);

        // for(int i = 0; i < N; i++)
        //     //cerr << hX[i] << ' ' << hB[i] << ' ' << hB1[i] << endl;
        //     cerr << hB[i] - hB1[i] << ' ';
        // cerr << endl;

        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;

            double e = norm_check_general<T>('F', N, 1, ldb, hB.data(), hB1.data());
            unit_check_error(e, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
