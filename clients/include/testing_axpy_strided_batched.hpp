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
hipblasStatus_t testing_axpy_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasAxpyStridedBatchedFn
        = FORTRAN ? hipblasAxpyStridedBatched<T, true> : hipblasAxpyStridedBatched<T, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;
    T      alpha        = argus.get_alpha<T>();

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    hipblasStride stridex = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX   = stridex * batch_count;
    size_t        sizeY   = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasAxpyStridedBatchedFn(
            handle, N, nullptr, nullptr, incx, stridex, nullptr, incy, stridey, batch_count));
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy_host(sizeY);
    host_vector<T> hy_device(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy_host(sizeY);
    device_vector<T> dy_device(sizeY);
    device_vector<T> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<T>(hy_host, 1, N, abs_incy, stridey, batch_count);
    hy_device = hy_host;

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_host, hy_host.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dy_device, hy_device.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasAxpyStridedBatchedFn(
            handle, N, d_alpha, dx, incx, stridex, dy_device, incy, stridey, batch_count));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasAxpyStridedBatchedFn(
            handle, N, &alpha, dx, incx, stridex, dy_host, incy, stridey, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hy_host.data(), dy_host, sizeof(T) * sizeY, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hy_device.data(), dy_device, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<T>(
                N, alpha, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stridex, hy_cpu.data(), hy_host.data());
            unit_check_general<T>(
                1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy_device.data());
        }
        if(argus.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, N, abs_incy, stridey, hy_cpu.data(), hy_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, N, abs_incy, stridey, hy_cpu.data(), hy_device.data(), batch_count);
        }
    } // end of if unit check

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

            CHECK_HIPBLAS_ERROR(hipblasAxpyStridedBatchedFn(
                handle, N, d_alpha, dx, incx, stridex, dy_device, incy, stridey, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_incy, e_stride_y, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            axpy_gflop_count<T>(N),
            axpy_gbyte_count<T>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
