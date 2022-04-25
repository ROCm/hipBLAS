/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_hbmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasHbmvFn = FORTRAN ? hipblasHbmv<T, true> : hipblasHbmv<T, false>;

    int N    = argus.N;
    int K    = argus.K;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasFillMode_t uplo = char2hipblas_fill(argus.uplo_option);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || K < 0 || lda <= K || !incx || !incy;
    if(invalid_size || !N)
    {
        hipblasStatus_t actual = hipblasHbmvFn(
            handle, uplo, N, K, nullptr, nullptr, lda, nullptr, incx, nullptr, nullptr, incy);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    size_t A_size = size_t(lda) * N;
    size_t x_size = size_t(N) * abs_incx;
    size_t y_size = size_t(N) * abs_incy;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hy(y_size);
    host_vector<T> hy_cpu(y_size);
    host_vector<T> hy_host(y_size);
    host_vector<T> hy_device(y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> dy(y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    // Initial Data on CPU
    hipblas_init_matrix(hA, argus, K, N, lda, 0, 1, hipblas_client_alpha_sets_nan, true, false);
    hipblas_init_vector(hx, argus, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, argus, N, abs_incy, 0, 1, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasHbmvFn(
            handle, uplo, N, K, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy, incy));

        CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * y_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasHbmvFn(handle, uplo, N, K, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));

        CHECK_HIP_ERROR(hipMemcpy(hy_device.data(), dy, sizeof(T) * y_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_hbmv<T>(
            uplo, N, K, h_alpha, hA.data(), lda, hx.data(), incx, h_beta, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, abs_incy, hy_cpu, hy_host);
            unit_check_general<T>(1, N, abs_incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu, hy_host);
            hipblas_error_device = norm_check_general<T>('F', 1, N, abs_incy, hy_cpu, hy_device);
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasHbmvFn(handle, uplo, N, K, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_K, e_alpha, e_lda, e_incx, e_beta, e_incy>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            hbmv_gflop_count<T>(N, K),
            hbmv_gbyte_count<T>(N, K),
            hipblas_error_host,
            hipblas_error_device);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
