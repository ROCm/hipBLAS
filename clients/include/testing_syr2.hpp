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
hipblasStatus_t testing_syr2(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasSyr2Fn = FORTRAN ? hipblasSyr2<T, true> : hipblasSyr2<T, false>;

    int               N         = argus.N;
    int               incx      = argus.incx;
    int               incy      = argus.incy;
    int               lda       = argus.lda;
    char              char_uplo = argus.uplo_option;
    hipblasFillMode_t uplo      = char2hipblas_fill(char_uplo);

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;
    int A_size   = lda * N;
    int x_size   = abs_incx * N;
    int y_size   = abs_incy * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || lda < 1 || lda < N || incx == 0 || incy == 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(incx < 0 || incy < 0)
        return HIPBLAS_STATUS_SUCCESS;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hy(y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> dy(y_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = argus.get_alpha<T>();

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, N, N, lda);
    hipblas_init<T>(hx, 1, N, abs_incx);
    hipblas_init<T>(hy, 1, N, abs_incy);
    hA_cpu = hA;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasSyr2Fn(handle, uplo, N, (T*)&alpha, dx, incx, dy, incy, dA, lda);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hA.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_syr2<T>(uplo, N, alpha, hx.data(), incx, hy.data(), incy, hA_cpu.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, lda, hA.data(), hA_cpu.data());
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
