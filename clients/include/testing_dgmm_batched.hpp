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

    int A_size = size_t(lda) * N;
    int C_size = size_t(ldc) * N;
    int k      = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    int X_size = size_t(incx) * k;
    if(!X_size)
        X_size = 1;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hA_copy[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_copy[batch_count];
    host_vector<T> hC[batch_count];
    host_vector<T> hC_1[batch_count];
    host_vector<T> hC_gold[batch_count];

    device_batch_vector<T> bA(batch_count, A_size);
    device_batch_vector<T> bx(batch_count, X_size);
    device_batch_vector<T> bC(batch_count, C_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dC(batch_count);

    int last = batch_count - 1;
    if(!dA || !dx || !dC || !bA[last] || !bx[last] || !bC[last])
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]      = host_vector<T>(A_size);
        hA_copy[b] = host_vector<T>(A_size);
        hx[b]      = host_vector<T>(X_size);
        hx_copy[b] = host_vector<T>(X_size);
        hC[b]      = host_vector<T>(C_size);
        hC_1[b]    = host_vector<T>(C_size);
        hC_gold[b] = host_vector<T>(C_size);

        srand(1);
        hipblas_init<T>(hA[b], M, N, lda);
        hipblas_init<T>(hx[b], 1, k, incx);
        hipblas_init<T>(hC[b], M, N, ldc);

        hA_copy[b] = hA[b];
        hx_copy[b] = hx[b];
        hC_1[b]    = hC[b];
        hC_gold[b] = hC[b];

        hipMemcpy(bA[b], hA[b].data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
        hipMemcpy(bx[b], hx[b].data(), sizeof(T) * X_size, hipMemcpyHostToDevice);
        hipMemcpy(bC[b], hC[b].data(), sizeof(T) * C_size, hipMemcpyHostToDevice);
    }

    hipMemcpy(dA, bA, sizeof(T*) * batch_count, hipMemcpyHostToDevice);
    hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice);
    hipMemcpy(dC, bC, sizeof(T*) * batch_count, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status = hipblasDgmmBatchedFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
        hipMemcpy(hC_1[b].data(), bC[b], sizeof(T) * C_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
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
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
