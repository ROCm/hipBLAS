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

template <typename T, typename U = T, typename V = T>
hipblasStatus_t testing_rot_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotBatchedFn
        = FORTRAN ? hipblasRotBatched<T, U, V, true> : hipblasRotBatched<T, U, V, false>;

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<U>       dc(1);
    device_vector<V>       ds(1);

    // Initial Data on CPU
    host_batch_vector<T> hx_host(N, incx, batch_count);
    host_batch_vector<T> hy_host(N, incy, batch_count);
    host_batch_vector<T> hx_device(N, incx, batch_count);
    host_batch_vector<T> hy_device(N, incy, batch_count);
    host_batch_vector<T> hx_cpu(N, incx, batch_count);
    host_batch_vector<T> hy_cpu(N, incy, batch_count);
    host_vector<U>       hc(1);
    host_vector<V>       hs(1);

    hipblas_init(hx_host, true);
    hipblas_init(hy_host, false);
    hx_device.copy_from(hx_host);
    hx_cpu.copy_from(hx_host);
    hy_device.copy_from(hy_host);
    hy_cpu.copy_from(hy_host);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);
    hipblas_init<int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // Test host
    CHECK_HIP_ERROR(dx.transfer_from(hx_host));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR((hipblasRotBatchedFn(
        handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, hc, hs, batch_count)));
    CHECK_HIP_ERROR(hx_host.transfer_from(dx));
    CHECK_HIP_ERROR(hy_host.transfer_from(dy));

    // Test device
    CHECK_HIP_ERROR(dx.transfer_from(hx_device));
    CHECK_HIP_ERROR(dy.transfer_from(hy_device));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR((hipblasRotBatchedFn(
        handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, dc, ds, batch_count)));
    CHECK_HIP_ERROR(hx_device.transfer_from(dx));
    CHECK_HIP_ERROR(hy_device.transfer_from(dy));

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<T, U, V>(N, hx_cpu[b], incx, hy_cpu[b], incy, *hc, *hs);
    }

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                near_check_general(1, N, incx, hx_cpu[b], hx_host[b], rel_error);
                near_check_general(1, N, incy, hy_cpu[b], hy_host[b], rel_error);
                near_check_general(1, N, incx, hx_cpu[b], hx_device[b], rel_error);
                near_check_general(1, N, incy, hy_cpu[b], hy_device[b], rel_error);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = std::max(norm_check_general<T>('F', 1, N, incx, hx_cpu, hx_host, batch_count),
                           norm_check_general<T>('F', 1, N, incy, hy_cpu, hy_host, batch_count));
            hipblas_error_device
                = std::max(norm_check_general<T>('F', 1, N, incx, hx_cpu, hx_device, batch_count),
                           norm_check_general<T>('F', 1, N, incy, hy_cpu, hy_device, batch_count));
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR((hipblasRotBatchedFn(handle,
                                                     N,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     dc,
                                                     ds,
                                                     batch_count)));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<T>(
            std::cout,
            arg,
            gpu_time_used,
            rot_gflop_count<T, T, U, V>(N),
            rot_gbyte_count<T>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
