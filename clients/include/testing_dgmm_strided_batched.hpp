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
hipblasStatus_t testing_dgmm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDgmmStridedBatchedFn
        = FORTRAN ? hipblasDgmmStridedBatched<T, true> : hipblasDgmmStridedBatched<T, false>;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);

    int M            = argus.M;
    int N            = argus.N;
    int lda          = argus.lda;
    int incx         = argus.incx;
    int ldc          = argus.ldc;
    int batch_count  = argus.batch_count;
    int stride_scale = argus.stride_scale;
    int k            = (side == HIPBLAS_SIDE_RIGHT ? N : M);

    int stride_A = size_t(lda) * N * stride_scale;
    int stride_x = size_t(incx) * k * stride_scale;
    int stride_C = size_t(ldc) * N * stride_scale;
    if(!stride_x)
        stride_x = 1;

    int A_size = stride_A * batch_count;
    int C_size = stride_C * batch_count;
    int X_size = stride_x * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < M || ldc < M || batch_count < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_copy(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hx_copy(X_size);
    host_vector<T> hC(C_size);
    host_vector<T> hC_1(C_size);
    host_vector<T> hC_gold(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dC(C_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda, stride_A, batch_count);
    hipblas_init<T>(hx, 1, k, incx, stride_x, batch_count);
    hipblas_init<T>(hC, M, N, ldc, stride_C, batch_count);
    hA_copy = hA;
    hx_copy = hx;
    hC_1    = hC;
    hC_gold = hC;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status = hipblasDgmmStridedBatchedFn(
        handle, side, M, N, dA, lda, stride_A, dx, incx, stride_x, dC, ldc, stride_C, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    hipMemcpy(hC_1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int b = 0; b < batch_count; b++)
        {
            auto hC_goldb = hC_gold + b * stride_C;
            auto hA_copyb = hA_copy + b * stride_A;
            auto hx_copyb = hx_copy + b * stride_x;
            for(size_t i1 = 0; i1 < M; i1++)
            {
                for(size_t i2 = 0; i2 < N; i2++)
                {
                    if(HIPBLAS_SIDE_RIGHT == side)
                    {
                        hC_goldb[i1 + i2 * ldc] = hA_copyb[i1 + i2 * lda] * hx_copyb[i2 * incx];
                    }
                    else
                    {
                        hC_goldb[i1 + i2 * ldc] = hA_copyb[i1 + i2 * lda] * hx_copyb[i1 * incx];
                    }
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_gold, hC_1);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
