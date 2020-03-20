/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

template <typename T>
hipblasStatus_t testing_geqrf_strided_batched(Arguments argus)
{
    int    M            = argus.M;
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
    host_vector<T> hA(A_size);
    host_vector<T> hA1(A_size);
    host_vector<T> hIpiv(Ipiv_size);
    host_vector<T> hIpiv1(Ipiv_size);

    device_vector<T> dA(A_size);
    device_vector<T> dIpiv(Ipiv_size);

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
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGeqrfStridedBatched<T>(
        handle, M, N, dA, lda, strideA, dIpiv, strideP, batch_count);

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        host_vector<T> work(N);
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geqrf(
                M, N, hA.data() + b * strideA, lda, hIpiv.data() + b * strideP, work.data(), N);

            if(argus.unit_check)
            {
                unit_check_general<T>(M, N, lda, hA.data() + b * strideA, hA1.data() + b * strideA);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
