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
hipblasStatus_t testing_dgmm(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasDgmmFn = FORTRAN ? hipblasDgmm<T, true> : hipblasDgmm<T, false>;

    hipblasSideMode_t side = char2hipblas_side(argus.side_option);

    int M    = argus.M;
    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int ldc  = argus.ldc;

    int A_size = size_t(lda) * N;
    int C_size = size_t(ldc) * N;
    int k      = (side == HIPBLAS_SIDE_RIGHT ? N : M);
    int X_size = size_t(incx) * k;
    if(!X_size)
        X_size = 1;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < M || ldc < M)
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
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hx, 1, k, incx);
    hipblas_init<T>(hC, M, N, ldc);
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
    status = hipblasDgmmFn(handle, side, M, N, dA, lda, dx, incx, dC, ldc);

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
        for(size_t i1 = 0; i1 < M; i1++)
        {
            for(size_t i2 = 0; i2 < N; i2++)
            {
                if(HIPBLAS_SIDE_RIGHT == side)
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[i2 * incx];
                }
                else
                {
                    hC_gold[i1 + i2 * ldc] = hA_copy[i1 + i2 * lda] * hx_copy[i1 * incx];
                }
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_gold, hC_1);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
