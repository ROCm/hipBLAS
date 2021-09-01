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
hipblasStatus_t testing_geqrf_batched(const Arguments& argus)
{
    using U      = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasGeqrfBatchedFn
        = FORTRAN ? hipblasGeqrfBatched<T, true> : hipblasGeqrfBatched<T, false>;

    int M           = argus.M;
    int N           = argus.N;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    size_t A_size    = size_t(lda) * N;
    int    Ipiv_size = min(M, N);

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
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA1(A_size, 1, batch_count);
    host_batch_vector<T> hIpiv(Ipiv_size, 1, batch_count);
    host_batch_vector<T> hIpiv1(Ipiv_size, 1, batch_count);
    int                  info;

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dIpiv(Ipiv_size, 1, batch_count);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial hA on CPU
    hipblas_init(hA, true);
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
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
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    CHECK_HIPBLAS_ERROR(hipblasGeqrfBatchedFn(
        handle, M, N, dA.ptr_on_device(), lda, dIpiv.ptr_on_device(), &info, batch_count));

    CHECK_HIP_ERROR(hIpiv1.transfer_from(dIpiv));
    CHECK_HIP_ERROR(hA1.transfer_from(dA));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        // Workspace query
        host_vector<T> work(1);
        cblas_geqrf(M, N, hA[0], lda, hIpiv[0], work.data(), -1);
        int lwork = type2int(work[0]);

        // Perform factorization
        work = host_vector<T>(lwork);
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geqrf(M, N, hA[b], lda, hIpiv[b], work.data(), N);
        }

        double e1 = norm_check_general<T>('F', M, N, lda, hA, hA1, batch_count);
        double e2 = norm_check_general<T>('F', min(M, N), 1, min(M, N), hIpiv, hIpiv1, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasGeqrfBatchedFn(
                handle, M, N, dA.ptr_on_device(), lda, dIpiv.ptr_on_device(), &info, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_lda, e_batch_count>{}.log_args<T>(std::cout,
                                                                    argus,
                                                                    gpu_time_used,
                                                                    geqrf_gflop_count<T>(N),
                                                                    geqrf_gbyte_count<T>(N),
                                                                    hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
