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
hipblasStatus_t testing_geqrf(const Arguments& argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasGeqrfFn = FORTRAN ? hipblasGeqrf<T, true> : hipblasGeqrf<T, false>;

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
    host_vector<T> hA(A_size);
    host_vector<T> hA1(A_size);
    host_vector<T> hIpiv(Ipiv_size);
    host_vector<T> hIpiv1(Ipiv_size);
    int            info;

    device_vector<T> dA(A_size);
    device_vector<T> dIpiv(Ipiv_size);

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
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, &info);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1.data(), dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        cblas_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        cblas_geqrf(M, N, hA.data(), lda, hIpiv.data(), work.data(), lwork);

        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            double e1 = norm_check_general<T>('F', M, N, lda, hA.data(), hA1.data());
            unit_check_error(e1, tolerance);

            double e2
                = norm_check_general<T>('F', min(M, N), 1, min(M, N), hIpiv.data(), hIpiv1.data());
            unit_check_error(e2, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
