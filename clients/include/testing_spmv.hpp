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
hipblasStatus_t testing_spmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasSpmvFn = FORTRAN ? hipblasSpmv<T, true> : hipblasSpmv<T, false>;

    int M    = argus.M;
    int incx = argus.incx;
    int incy = argus.incy;

    size_t A_size = M * (M + 1) / 2;

    hipblasFillMode_t uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasStatus_t   status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx == 0 || incy == 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(M * incx);
    host_vector<T> hy(M * incy);
    host_vector<T> hy_cpu(M * incy);
    host_vector<T> hy_host(M * incy);
    host_vector<T> hy_device(M * incy);

    device_vector<T> dA(A_size);
    device_vector<T> dx(M * incx);
    device_vector<T> dy(M * incy);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, 1, A_size, 1);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, M, incy);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * M * incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * M * incy, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasSpmvFn(handle, uplo, M, &h_alpha, dA, dx, incx, &h_beta, dy, incy));

    CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * M * incy, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * M * incy, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasSpmvFn(handle, uplo, M, d_alpha, dA, dx, incx, d_beta, dy, incy));

    CHECK_HIP_ERROR(hipMemcpy(hy_device.data(), dy, sizeof(T) * M * incy, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_spmv<T>(uplo, M, h_alpha, hA.data(), hx.data(), incx, h_beta, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy_cpu, hy_host);
            unit_check_general<T>(1, M, incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', 1, M, incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<T>('F', 1, M, incy, hy_cpu, hy_device);
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * M * incy, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasSpmvFn(handle, uplo, M, d_alpha, dA, dx, incx, d_beta, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_alpha, e_incx, e_beta, e_incy>{}.log_args<T>(std::cout,
                                                                          argus,
                                                                          gpu_time_used,
                                                                          spmv_gflop_count<T>(M),
                                                                          spmv_gbyte_count<T>(M),
                                                                          hipblas_error_host,
                                                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
