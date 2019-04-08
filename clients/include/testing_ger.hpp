/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
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

template <typename T>
hipblasStatus_t testing_ger(Arguments argus)
{

    int M    = argus.M;
    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;
    int lda  = argus.lda;

    int A_size = lda * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    if(N < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(lda < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incx <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incy <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(A_size);
    vector<T> hx(M * incx);
    vector<T> hy(N * incy);

    T *dA, *dx, *dy;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;

    hipblasHandle_t handle;

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dx, M * incx * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dy, N * incy * sizeof(T)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, N, incy);

    // copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    hB = hA;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * M * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * N * incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {

        status = hipblasGer<T>(handle, M, N, (T*)&alpha, dx, incx, dy, incy, dA, lda);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            CHECK_HIP_ERROR(hipFree(dA));
            CHECK_HIP_ERROR(hipFree(dx));
            CHECK_HIP_ERROR(hipFree(dy));
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
        cblas_ger<T>(M, N, alpha, hx.data(), incx, hy.data(), incy, hB.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, lda, hB.data(), hA.data());
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
