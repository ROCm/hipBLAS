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
hipblasStatus_t testing_geqrf_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGeqrfBatchedFn
        = FORTRAN ? hipblasGeqrfBatched<T, true> : hipblasGeqrfBatched<T, false>;

    int M           = argus.M;
    int N           = argus.N;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    int A_size    = lda * N;
    int Ipiv_size = min(M, N);

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
    host_vector<T> hA[batch_count];
    host_vector<T> hA1[batch_count];
    host_vector<T> hIpiv[batch_count];
    host_vector<T> hIpiv1[batch_count];
    int            info;

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bIpiv(batch_count, Ipiv_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dIpiv(batch_count);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]     = host_vector<T>(A_size);
        hA1[b]    = host_vector<T>(A_size);
        hIpiv[b]  = host_vector<T>(Ipiv_size);
        hIpiv1[b] = host_vector<T>(Ipiv_size);

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

        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b].data(), A_size * sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(bIpiv[b], hIpiv[b].data(), Ipiv_size * sizeof(T), hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dA, bA, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, bIpiv, batch_count * sizeof(T*), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGeqrfBatchedFn(handle, M, N, dA, lda, dIpiv, &info, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // Copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(hipMemcpy(hA1[b].data(), bA[b], A_size * sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv1[b].data(), bIpiv[b], Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        cblas_geqrf(M, N, hA[0].data(), lda, hIpiv[0].data(), work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geqrf(M, N, hA[b].data(), lda, hIpiv[b].data(), work.data(), N);

            if(argus.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = eps * 2000;

                double e1 = norm_check_general<T>('F', M, N, lda, hA[b].data(), hA1[b].data());
                unit_check_error(e1, tolerance);

                double e2 = norm_check_general<T>(
                    'F', min(M, N), 1, min(M, N), hIpiv[b].data(), hIpiv1[b].data());
                unit_check_error(e2, tolerance);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
