/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

template <typename T>
hipblasStatus_t testing_geqrf(const Arguments& argus)
{
    using U             = real_t<T>;
    bool FORTRAN        = argus.fortran;
    auto hipblasGeqrfFn = FORTRAN ? hipblasGeqrf<T, true> : hipblasGeqrf<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int K   = std::min(M, N);
    int lda = argus.lda;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = K;

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
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(dIpiv, 0, Ipiv_size * sizeof(T)));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, &info));

    // Copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA1, dA, A_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hIpiv1, dIpiv, Ipiv_size * sizeof(T), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
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

        double e1     = norm_check_general<T>('F', M, N, lda, hA, hA1);
        double e2     = norm_check_general<T>('F', K, 1, K, hIpiv, hIpiv1);
        hipblas_error = e1 + e2;

        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = eps * 2000;

            unit_check_error(e1, tolerance);
            unit_check_error(e2, tolerance);
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

            CHECK_HIPBLAS_ERROR(hipblasGeqrfFn(handle, M, N, dA, lda, dIpiv, &info));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_lda>{}.log_args<T>(std::cout,
                                                     argus,
                                                     gpu_time_used,
                                                     geqrf_gflop_count<T>(N, M),
                                                     ArgumentLogging::NA_value,
                                                     hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
