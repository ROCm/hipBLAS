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
hipblasStatus_t testing_getrf(const Arguments& argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasGetrfFn = FORTRAN ? hipblasGetrf<T, true> : hipblasGetrf<T, false>;

    int M   = argus.N;
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
    host_vector<T>   hA(A_size);
    host_vector<T>   hA1(A_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    host_vector<int> hInfo(1);
    host_vector<int> hInfo1(1);

    device_vector<T>   dA(A_size);
    device_vector<int> dIpiv(Ipiv_size);
    device_vector<int> dInfo(1);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);

    // scale A to avoid singularities
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(i == j)
                hA[i + j * lda] += 400;
            else
                hA[i + j * lda] -= 4;
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, sizeof(int)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrfFn(handle, N, dA, lda, dIpiv, dInfo);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hInfo1.data(), dInfo, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        hInfo[0] = cblas_getrf(M, N, hA.data(), lda, hIpiv.data());

        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            double e = norm_check_general<T>('F', M, N, lda, hA.data(), hA1.data());
            unit_check_error(e, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
