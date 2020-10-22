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
hipblasStatus_t testing_getri_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGetriBatchedFn
        = FORTRAN ? hipblasGetriBatched<T, true> : hipblasGetriBatched<T, false>;

    int M           = argus.N;
    int N           = argus.N;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    int strideP   = min(M, N);
    int A_size    = lda * N;
    int Ipiv_size = strideP * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || lda < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T>   hA[batch_count];
    host_vector<T>   hA1[batch_count];
    host_vector<T>   hC[batch_count];
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    host_vector<int> hInfo(batch_count);
    host_vector<int> hInfo1(batch_count);

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bC(batch_count, A_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dC(batch_count);
    device_vector<int>      dIpiv(Ipiv_size);
    device_vector<int>      dInfo(batch_count);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]       = host_vector<T>(A_size);
        hA1[b]      = host_vector<T>(A_size);
        hC[b]       = host_vector<T>(A_size);
        int* hIpivb = hIpiv.data() + b * strideP;

        hipblas_init<T>(hA[b], M, N, lda);

        // scale A to avoid singularities
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }

        // perform LU factorization on A
        hInfo[b] = cblas_getrf(M, N, hA[b].data(), lda, hIpivb);

        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b].data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dA, bA, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, bC, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv, Ipiv_size * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, batch_count * sizeof(int)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetriBatchedFn(handle, N, dA, lda, dIpiv, dC, lda, dInfo, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // Copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        CHECK_HIP_ERROR(hipMemcpy(hA1[b].data(), bC[b], A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hInfo1.data(), dInfo, batch_count * sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            // Workspace query
            host_vector<T> work(1);
            cblas_getri(N, hA[b].data(), lda, hIpiv.data() + b * strideP, work.data(), -1);
            int lwork = type2int(work[0]);

            // Perform inversion
            work = host_vector<T>(lwork);
            hInfo[b]
                = cblas_getri(N, hA[b].data(), lda, hIpiv.data() + b * strideP, work.data(), lwork);

            if(argus.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = eps * 2000;

                double e = norm_check_general<T>('F', M, N, lda, hA[b].data(), hA1[b].data());
                unit_check_error(e, tolerance);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
