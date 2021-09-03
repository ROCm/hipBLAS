/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

template <typename T>
hipblasStatus_t testing_getrf(const Arguments& argus)
{
    using U             = real_t<T>;
    bool FORTRAN        = argus.fortran;
    auto hipblasGetrfFn = FORTRAN ? hipblasGetrf<T, true> : hipblasGetrf<T, false>;

    int M   = argus.N;
    int N   = argus.N;
    int lda = argus.lda;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = min(M, N);

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

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

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
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(dInfo, 0, sizeof(int)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasGetrfFn(handle, N, dA, lda, dIpiv, dInfo));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1, dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1, dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hInfo1, dInfo, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        hInfo[0] = cblas_getrf(M, N, hA.data(), lda, hIpiv.data());

        hipblas_error = norm_check_general<T>('F', M, N, lda, hA, hA1);
        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(hipblas_error, tolerance);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGetrfFn(handle, N, dA, lda, dIpiv, dInfo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_lda>{}.log_args<T>(std::cout,
                                                argus,
                                                gpu_time_used,
                                                getrf_gflop_count<T>(N, M),
                                                ArgumentLogging::NA_value,
                                                hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
