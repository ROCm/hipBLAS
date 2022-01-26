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
hipblasStatus_t testing_copy_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasCopyStridedBatchedFn
        = FORTRAN ? hipblasCopyStridedBatched<T, true> : hipblasCopyStridedBatched<T, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    int           abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey  = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX    = stridex * batch_count;
    size_t        sizeY    = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasCopyStridedBatchedFn(
            handle, N, nullptr, incx, stridex, nullptr, incy, stridey, batch_count));
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);

    double gpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<T>(hy, 1, N, abs_incy, stridey, batch_count);

    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasCopyStridedBatchedFn(
            handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /*=====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_copy<T>(N, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy.data());
        }
        if(argus.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stridey, hy_cpu, hy, batch_count);
        }

    } // end of if unit check

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasCopyStridedBatchedFn(
                handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_stride_x, e_stride_y, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            copy_gflop_count<T>(N),
            copy_gbyte_count<T>(N),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
