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
hipblasStatus_t testing_geqrf_batched(Arguments argus)
{
    int M           = argus.M;
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
    host_vector<T> hA[batch_count];
    host_vector<T> hA1[batch_count];
    host_vector<T> hIpiv(Ipiv_size);
    host_vector<T> hIpiv1(Ipiv_size);

    device_batch_vector<T> bA(batch_count, A_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T>        dIpiv(Ipiv_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]  = host_vector<T>(A_size);
        hA1[b] = host_vector<T>(A_size);

        hipblas_init<T>(hA[b], M, N, lda);

        // Copy data from CPU to device
        CHECK_HIP_ERROR(hipMemcpy(bA[b], hA[b].data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dA, bA, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGeqrfBatched<T>(handle, M, N, dA, lda, dIpiv, strideP, batch_count);

    // Copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        CHECK_HIP_ERROR(hipMemcpy(hA1[b].data(), bA[b], A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // cblas_geqrf(M, N, hA.data(), lda, hIpiv.data());

        // if(argus.unit_check)
        // {
        //     unit_check_general<T>(M, N, lda, hA.data(), hA1.data());
        // }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
