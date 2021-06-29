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

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_dgmm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDgmmBatchedFn
        = FORTRAN ? hipblasDgmmBatched<T, true> : hipblasDgmmBatched<T, false>;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);

    int M           = argus.M;
    int N           = argus.N;
    int lda         = argus.lda;
    int incx        = argus.incx;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    size_t A_size = size_t(lda) * N;
    size_t C_size = size_t(ldc) * N;
    int    k      = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA_copy(A_size, 1, batch_count);
    host_batch_vector<T> hx(k, incx, batch_count);
    host_batch_vector<T> hx_copy(k, incx, batch_count);
    host_batch_vector<T> hC(C_size, 1, batch_count);
    host_batch_vector<T> hC_1(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(k, incx, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    hipblas_init(hA, true);
    hipblas_init(hx);
    hipblas_init(hC);

    hA_copy.copy_from(hA);
    hx_copy.copy_from(hx);
    hC_1.copy_from(hC);
    hC_gold.copy_from(hC_gold);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasDgmmBatchedFn(handle,
                                             side,
                                             M,
                                             N,
                                             dA.ptr_on_device(),
                                             lda,
                                             dx.ptr_on_device(),
                                             incx,
                                             dC.ptr_on_device(),
                                             ldc,
                                             batch_count));
    CHECK_HIP_ERROR(hC_1.transfer_from(dC));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int b = 0; b < batch_count; b++)
        {
            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(HIPBLAS_SIDE_RIGHT == side)
                    {
                        hC_gold[b][i1 + i2 * ldc]
                            = hA_copy[b][i1 + i2 * lda] * hx_copy[b][i2 * incx];
                    }
                    else
                    {
                        hC_gold[b][i1 + i2 * ldc]
                            = hA_copy[b][i1 + i2 * lda] * hx_copy[b][i1 * incx];
                    }
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_gold, hC_1);
        }

        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', M, N, ldc, hC_gold, hC_1, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasDgmmBatchedFn(handle,
                                                     side,
                                                     M,
                                                     N,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_side_option, e_M, e_N, e_lda, e_incx, e_ldc, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            dgmm_gflop_count<T>(M, N),
            dgmm_gbyte_count<T>(M, N, k),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
