/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, typename U = T>
hipblasStatus_t testing_scal_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasScalStridedBatchedFn
        = FORTRAN ? hipblasScalStridedBatched<T, U, true> : hipblasScalStridedBatched<T, U, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;
    int    unit_check   = argus.unit_check;
    int    timing       = argus.timing;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    U alpha = argus.get_alpha<U>();

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(
            hipblasScalStridedBatchedFn(handle, N, nullptr, nullptr, incx, stridex, batch_count));
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hz(sizeX);

    device_vector<T> dx(sizeX);

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx, stridex, batch_count);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(
            hipblasScalStridedBatchedFn(handle, N, &alpha, dx, incx, stridex, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<T, U>(N, alpha, hz.data() + b * stridex, incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, stridex, hz, hx);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasScalStridedBatchedFn(handle, N, &alpha, dx, incx, stridex, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            scal_gflop_count<T, U>(N),
            scal_gbyte_count<T>(N),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
