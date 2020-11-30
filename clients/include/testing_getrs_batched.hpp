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
hipblasStatus_t testing_getrs_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGetrsBatchedFn
        = FORTRAN ? hipblasGetrsBatched<T, true> : hipblasGetrsBatched<T, false>;

    int N           = argus.N;
    int lda         = argus.lda;
    int ldb         = argus.ldb;
    int batch_count = argus.batch_count;

    int strideP   = N;
    int A_size    = lda * N;
    int B_size    = ldb * 1;
    int Ipiv_size = strideP * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA[batch_count];
    host_vector<T>   hX[batch_count];
    host_vector<T>   hB[batch_count];
    host_vector<T>   hB1[batch_count];
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    int              info;

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bB(batch_count, B_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<int>      dIpiv(Ipiv_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]  = host_vector<T>(A_size);
        hX[b]  = host_vector<T>(B_size);
        hB[b]  = host_vector<T>(B_size);
        hB1[b] = host_vector<T>(B_size);

        hipblas_init<T>(hA[b], N, N, lda);
        hipblas_init<T>(hX[b], N, 1, ldb);

        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }

        // Calculate hB = hA*hX;
        cblas_gemm<T>(
            op, op, N, 1, N, (T)1, hA[b].data(), lda, hX[b].data(), ldb, (T)0, hB[b].data(), ldb);

        // LU factorize hA on the CPU
        info = cblas_getrf<T>(N, N, hA[b].data(), lda, hIpiv.data() + b * strideP);
        if(info != 0)
        {
            cerr << "LU decomposition failed" << endl;
            return HIPBLAS_STATUS_SUCCESS;
        }

        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b].data(), A_size * sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB[b], hB[b].data(), B_size * sizeof(T), hipMemcpyHostToDevice));
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, bA, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, bB, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrsBatchedFn(handle, op, N, 1, dA, lda, dIpiv, dB, ldb, &info, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        CHECK_HIP_ERROR(hipMemcpy(hB1[b].data(), bB[b], B_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_getrs(
                'N', N, 1, hA[b].data(), lda, hIpiv.data() + b * strideP, hB[b].data(), ldb);

            if(argus.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = N * eps * 100;

                double e = norm_check_general<T>('F', N, 1, ldb, hB[b].data(), hB1[b].data());
                unit_check_error(e, tolerance);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
