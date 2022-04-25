/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

template <typename T>
hipblasStatus_t testing_getrs_batched(const Arguments& argus)
{
    using U      = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasGetrsBatchedFn
        = FORTRAN ? hipblasGetrsBatched<T, true> : hipblasGetrsBatched<T, false>;

    int N           = argus.N;
    int lda         = argus.lda;
    int ldb         = argus.ldb;
    int batch_count = argus.batch_count;

    hipblasStride strideP   = N;
    size_t        A_size    = size_t(lda) * N;
    size_t        B_size    = size_t(ldb) * 1;
    size_t        Ipiv_size = strideP * batch_count;

    // Check to prevent memory allocation error
    if(N < 0 || lda < N || ldb < N || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hX(B_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hB1(B_size, 1, batch_count);
    host_vector<int>     hIpiv(Ipiv_size);
    host_vector<int>     hIpiv1(Ipiv_size);
    int                  info;

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_vector<int>     dIpiv(Ipiv_size);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial hA, hB, hX on CPU
    hipblas_init(hA, true);
    hipblas_init(hX);
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        // scale A to avoid singularities
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                if(i == j)
                    hA[b][i + j * lda] += 400;
                else
                    hA[b][i + j * lda] -= 4;
            }
        }

        // Calculate hB = hA*hX;
        cblas_gemm<T>(op, op, N, 1, N, (T)1, hA[b], lda, hX[b], ldb, (T)0, hB[b], ldb);

        // LU factorize hA on the CPU
        info = cblas_getrf<T>(N, N, hA[b], lda, hIpiv.data() + b * strideP);
        if(info != 0)
        {
            std::cerr << "LU decomposition failed" << std::endl;
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                  op,
                                                  N,
                                                  1,
                                                  dA.ptr_on_device(),
                                                  lda,
                                                  dIpiv,
                                                  dB.ptr_on_device(),
                                                  ldb,
                                                  &info,
                                                  batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB1.transfer_from(dB));
        CHECK_HIP_ERROR(
            hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_getrs('N', N, 1, hA[b], lda, hIpiv.data() + b * strideP, hB[b], ldb);
        }

        hipblas_error = norm_check_general<T>('F', N, 1, ldb, hB, hB1, batch_count);
        if(argus.unit_check)
        {
            U      eps       = std::numeric_limits<U>::epsilon();
            double tolerance = N * eps * 100;

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

            CHECK_HIPBLAS_ERROR(hipblasGetrsBatchedFn(handle,
                                                      op,
                                                      N,
                                                      1,
                                                      dA.ptr_on_device(),
                                                      lda,
                                                      dIpiv,
                                                      dB.ptr_on_device(),
                                                      ldb,
                                                      &info,
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_lda, e_ldb, e_batch_count>{}.log_args<T>(std::cout,
                                                                      argus,
                                                                      gpu_time_used,
                                                                      getrs_gflop_count<T>(N, 1),
                                                                      ArgumentLogging::NA_value,
                                                                      hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
