/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

template <typename T, typename U>
hipblasStatus_t testing_getrs_strided_batched(Arguments argus)
{
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    int    batch_count  = argus.batch_count;
    double stride_scale = argus.stride_scale;

    int strideA   = lda * N * stride_scale;
    int strideB   = ldb * 1 * stride_scale;
    int strideP   = N * stride_scale;
    int A_size    = strideA * batch_count;
    int B_size    = strideB * batch_count;
    int Ipiv_size = strideP * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

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
    host_vector<T>   hA(A_size);
    host_vector<T>   hX(B_size);
    host_vector<T>   hB(B_size);
    host_vector<T>   hB1(B_size);
    host_vector<int> hIpiv(Ipiv_size);
    host_vector<int> hIpiv1(Ipiv_size);
    int              info;

    device_vector<T, 1>   dA(A_size);
    device_vector<T, 1>   dB(B_size);
    device_vector<int, 1> dIpiv(Ipiv_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial hA, hB, hX on CPU
    srand(1);
    hipblasOperation_t op = HIPBLAS_OP_N;
    for(int b = 0; b < batch_count; b++)
    {
        T*   hAb    = hA.data() + b * strideA;
        T*   hXb    = hX.data() + b * strideB;
        T*   hBb    = hB.data() + b * strideB;
        int* hIpivb = hIpiv.data() + b * strideP;

        hipblas_init<T>(hAb, N, N, lda);
        hipblas_init<T>(hXb, N, 1, ldb);

        // Put hA entries into range [0, 1], make diagonally dominant
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < N; j++)
            {
                hAb[i + j * lda] = (hAb[i + j * lda] - 1.0) / 10.0;

                if(i == j)
                    hAb[i + j * lda] *= 100;
            }
        }

        // Calculate hB = hA*hX;
        cblas_gemm<T>(op, op, N, 1, N, 1, hAb, lda, hXb, ldb, 0, hBb, ldb);

        // LU factorize hA on the CPU
        info = cblas_getrf<T>(N, N, hAb, lda, hIpivb);
        if(info != 0)
        {
            cerr << "LU decomposition failed" << endl;
            return HIPBLAS_STATUS_SUCCESS;
        }
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), A_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), B_size * sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dIpiv, hIpiv.data(), Ipiv_size * sizeof(int), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */

    status = hipblasGetrsStridedBatched<T>(
        handle, op, N, 1, dA, lda, strideA, dIpiv, strideP, dB, ldb, strideB, &info, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hB1.data(), dB, B_size * sizeof(T), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hIpiv1.data(), dIpiv, Ipiv_size * sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU LAPACK
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_getrs('N',
                        N,
                        1,
                        hA.data() + b * strideA,
                        lda,
                        hIpiv.data() + b * strideP,
                        hB.data() + b * strideB,
                        ldb);

            if(argus.unit_check)
            {
                U      eps       = std::numeric_limits<U>::epsilon();
                double tolerance = N * eps * 100;

                double e = norm_check_general<T>(
                    'F', N, 1, ldb, hB.data() + b * strideB, hB1.data() + b * strideB);
                unit_check_error(e, tolerance);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
