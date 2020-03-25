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

/* ============================================================================================ */

template <typename T, typename U>
hipblasStatus_t testing_her(Arguments argus)
{
    int N    = argus.N;
    int incx = argus.incx;
    int lda  = argus.lda;

    int               A_size = lda * N;
    hipblasFillMode_t uplo   = char2hipblas_fill(argus.uplo_option);

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || lda < N || incx == 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(A_size);
    vector<T> hx(N * incx);

    T *dA, *dx;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    U alpha = (U)argus.alpha;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, N * incx * sizeof(T)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, N, lda);
    hipblas_init<T>(hx, 1, N, incx);

    // copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    hB = hA;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {

        status = hipblasHer<T>(handle, uplo, N, (U*)&alpha, dx, incx, dA, lda);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            CHECK_HIP_ERROR(hipFree(dA));
            CHECK_HIP_ERROR(hipFree(dx));
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hA.data(), dA, sizeof(T) * N * lda, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_her<T>(uplo, N, alpha, hx.data(), incx, hB.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, lda, hB.data(), hA.data());
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
