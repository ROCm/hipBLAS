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
hipblasStatus_t testing_syr(const Arguments& argus)
{
    bool FORTRAN      = argus.fortran;
    auto hipblasSyrFn = FORTRAN ? hipblasSyr<T, true> : hipblasSyr<T, false>;

    int               N         = argus.N;
    int               incx      = argus.incx;
    int               lda       = argus.lda;
    char              char_uplo = argus.uplo_option;
    hipblasFillMode_t uplo      = char2hipblas_fill(char_uplo);

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t A_size   = size_t(lda) * N;
    size_t x_size   = abs_incx * size_t(N);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || lda < N || lda < 1;
    if(invalid_size || !N)
    {
        hipblasStatus_t actual
            = hipblasSyrFn(handle, uplo, N, nullptr, nullptr, incx, nullptr, lda);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hA_host(A_size);
    host_vector<T> hA_device(A_size);
    host_vector<T> hx(x_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> d_alpha(1);

    T h_alpha = argus.get_alpha<T>();

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(hA, argus, N, N, lda, 0, 1, hipblas_client_never_set_nan, true, false);
    hipblas_init_vector(hx, argus, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasSyrFn(handle, uplo, N, &h_alpha, dx, incx, dA, lda));

        CHECK_HIP_ERROR(hipMemcpy(hA_host.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasSyrFn(handle, uplo, N, d_alpha, dx, incx, dA, lda));

        CHECK_HIP_ERROR(hipMemcpy(hA_device.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_syr<T>(uplo, N, h_alpha, hx.data(), incx, hA_cpu.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_host.data());
            unit_check_general<T>(N, N, lda, hA_cpu.data(), hA_device.data());
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, lda, hA_cpu.data(), hA_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', N, N, lda, hA_cpu.data(), hA_device.data());
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSyrFn(handle, uplo, N, d_alpha, dx, incx, dA, lda));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_incx, e_lda>{}.log_args<T>(std::cout,
                                                                 argus,
                                                                 gpu_time_used,
                                                                 syr_gflop_count<T>(N),
                                                                 syr_gbyte_count<T>(N),
                                                                 hipblas_error_host,
                                                                 hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
