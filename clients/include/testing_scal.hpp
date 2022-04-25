/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T, typename U = T>
hipblasStatus_t testing_scal(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasScalFn = FORTRAN ? hipblasScal<T, U, true> : hipblasScal<T, U, false>;

    int N          = argus.N;
    int incx       = argus.incx;
    int unit_check = argus.unit_check;
    int timing     = argus.timing;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasScalFn(handle, N, nullptr, nullptr, incx));
        return HIPBLAS_STATUS_SUCCESS;
    }

    size_t sizeX = size_t(N) * incx;
    U      alpha = argus.get_alpha<U>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T>   hx(sizeX);
    host_vector<T>   hz(sizeX);
    device_vector<T> dx(sizeX);

    double gpu_time_used, cpu_time_used;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(hx, argus, N, incx, 0, 1, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasScalFn(handle, N, &alpha, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_scal<T, U>(N, alpha, hz.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, incx, hz.data(), hx.data());
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general('F', 1, N, incx, hz.data(), hx.data());
        }

    } // end of if unit check

    //  BLAS_1_RESULT_PRINT

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasScalFn(handle, N, &alpha, dx, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx>{}.log_args<T>(std::cout,
                                                 argus,
                                                 gpu_time_used,
                                                 scal_gflop_count<T, U>(N),
                                                 scal_gbyte_count<T>(N),
                                                 hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
