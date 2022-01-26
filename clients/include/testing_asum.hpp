/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_asum(const Arguments& argus)
{
    using Tr           = real_t<T>;
    bool FORTRAN       = argus.fortran;
    auto hipblasAsumFn = FORTRAN ? hipblasAsum<T, Tr, true> : hipblasAsum<T, Tr, false>;

    int N    = argus.N;
    int incx = argus.incx;

    hipblasLocalHandle handle(argus);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(1);
        host_vector<Tr>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasAsumFn(handle, N, nullptr, incx, d_hipblas_result_0));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);

        return HIPBLAS_STATUS_SUCCESS;
    }

    size_t sizeX = size_t(N) * incx;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);

    device_vector<T>  dx(sizeX);
    device_vector<Tr> d_hipblas_result(1);
    Tr                cpu_result, hipblas_result_host, hipblas_result_device;

    double gpu_time_used, hipblas_error_host = 0, hipblas_error_device = 0;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * N * incx, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // hipblasAsum accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasAsumFn(handle, N, dx, incx, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasAsumFn(handle, N, dx, incx, &hipblas_result_host));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        cblas_asum<T, Tr>(N, hx.data(), incx, &cpu_result);

        if(argus.unit_check)
        {
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('M', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('M', 1, 1, 1, &cpu_result, &hipblas_result_device);
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasAsumFn(handle, N, dx, incx, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(std::cout,
                                                 argus,
                                                 gpu_time_used,
                                                 asum_gflop_count<T>(N),
                                                 asum_gbyte_count<T>(N),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
