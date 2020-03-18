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
hipblasStatus_t testing_getrf(Arguments argus)
{
    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;

    int A_size    = lda * N;
    int Ipiv_size = min(M, N);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // Check to prevent memory allocation error
    if(M < 0 || N < 0 || lda < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T>   hA(A_size, 0);
    vector<T>   hA1(A_size, 0);
    vector<int> hIpiv(Ipiv_size, 0);
    vector<int> hIpiv1(Ipiv_size, 0);
    int         hInfo;
    int         hInfo1;

    T*   dA;
    int* dIpiv;
    int* dInfo;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dIpiv, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMalloc(&dInfo, sizeof(int)));

    // Initial hA on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, sizeof(int)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrf<T>(handle, M, N, dA, lda, dIpiv, dInfo);

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&hInfo1, dInfo, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        hInfo = cblas_getrf(M, N, hA.data(), lda, hIpiv.data());

        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, lda, hA.data(), hA1.data());
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dIpiv));
    CHECK_HIP_ERROR(hipFree(dInfo));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
