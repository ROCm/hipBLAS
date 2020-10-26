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
hipblasStatus_t testing_getrf_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGetrfStridedBatchedFn
        = FORTRAN ? hipblasGetrfStridedBatched<T, true> : hipblasGetrfStridedBatched<T, false>;

    int    M            = argus.N;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    batch_count  = argus.batch_count;
    double stride_scale = argus.stride_scale;

    int strideA   = lda * N * stride_scale;
    int strideP   = min(M, N) * stride_scale;
    int A_size    = strideA * batch_count;
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
    host_vector<T>   hA(A_size);
    host_vector<T>   hA1(A_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    host_vector<int> hInfo(batch_count);
    host_vector<int> hInfo1(batch_count);

    device_vector<T>   dA(A_size);
    device_vector<int> dIpiv(Ipiv_size);
    device_vector<int> dInfo(batch_count);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        T* hAb = hA.data() + b * strideA;

        hipblas_init<T>(hAb, M, N, lda);

        // scale A to avoid singularities
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hAb[i + j * lda] += 400;
                else
                    hAb[i + j * lda] -= 4;
            }
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, batch_count * sizeof(int)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrfStridedBatchedFn(
        handle, N, dA, lda, strideA, dIpiv, strideP, dInfo, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
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
            hInfo[b] = cblas_getrf(M, N, hA.data() + b * strideA, lda, hIpiv.data() + b * strideP);

            if(argus.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = eps * 2000;

                double e = norm_check_general<T>(
                    'F', M, N, lda, hA.data() + b * strideA, hA1.data() + b * strideA);
                unit_check_error(e, tolerance);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
