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
hipblasStatus_t testing_rot(const Arguments& arg)
{
    bool FORTRAN      = arg.fortran;
    auto hipblasRotFn = FORTRAN ? hipblasRot<T, U, V, true> : hipblasRot<T, U, V, false>;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<U> dc(1);
    device_vector<V> ds(1);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<U> hc(1);
    host_vector<V> hs(1);
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);
    hipblas_init<int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;

    cblas_rot<T, U, V>(N, cx.data(), incx, cy.data(), incy, *hc, *hs);

    if(arg.unit_check || arg.norm_check)
    {
        // Test host
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_HIPBLAS_ERROR(hipblasRotFn(handle, N, dx, incx, dy, incy, hc, hs));

            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, incy, cy.data(), ry.data(), double(rel_error));
            }
            if(arg.norm_check)
            {
                hipblas_error_host = norm_check_general<T>('F', 1, N, incx, cx, rx);
                hipblas_error_host += norm_check_general<T>('F', 1, N, incy, cy, ry);
            }
        }

        // Test device
        {
            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            CHECK_HIPBLAS_ERROR(hipblasRotFn(handle, N, dx, incx, dy, incy, dc, ds));
            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, incy, cy.data(), ry.data(), double(rel_error));
            }
            if(arg.norm_check)
            {
                hipblas_error_device = norm_check_general<T>('F', 1, N, incx, cx, rx);
                hipblas_error_device += norm_check_general<T>('F', 1, N, incy, cy, ry);
            }
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasRotFn(handle, N, dx, incx, dy, incy, dc, ds));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<T>(std::cout,
                                                         arg,
                                                         gpu_time_used,
                                                         rot_gflop_count<T, T, U, V>(N),
                                                         rot_gbyte_count<T>(N),
                                                         hipblas_error_host,
                                                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
