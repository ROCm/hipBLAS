/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, typename U = T, typename V = T>
hipblasStatus_t testing_rot_strided_batched(const Arguments& arg)
{
    bool FORTRAN                    = arg.fortran;
    auto hipblasRotStridedBatchedFn = FORTRAN ? hipblasRotStridedBatched<T, U, V, true>
                                              : hipblasRotStridedBatched<T, U, V, false>;

    int    N            = arg.N;
    int    incx         = arg.incx;
    int    incy         = arg.incy;
    double stride_scale = arg.stride_scale;
    int    stride_x     = N * incx * stride_scale;
    int    stride_y     = N * incy * stride_scale;
    int    batch_count  = arg.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    else if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_x = N * size_t(incx) + size_t(stride_x) * size_t(batch_count - 1);
    size_t size_y = N * size_t(incy) + size_t(stride_y) * size_t(batch_count - 1);

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
    hipblas_init<T>(hx, 1, N, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, N, incy, stride_y, batch_count);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);
    hipblas_init<int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx = hx;
    host_vector<T> cy = hy;
    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<T, U, V>(
            N, cx.data() + b * stride_x, incx, cy.data() + b * stride_y, incy, *hc, *hs);
    }

    if(arg.unit_check)
    {
        // Test host
        {
            status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            status_2 = ((hipblasRotStridedBatchedFn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, hc, hs, batch_count)));

            if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_1 != HIPBLAS_STATUS_SUCCESS)
                    return status_1;
                if(status_2 != HIPBLAS_STATUS_SUCCESS)
                    return status_2;
            }
            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general<T>(1, N, batch_count, incx, stride_x, cx, rx, rel_error);
                near_check_general<T>(1, N, batch_count, incy, stride_y, cy, ry, rel_error);
            }
        }

        // Test device
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));
            status_4 = ((hipblasRotStridedBatchedFn(
                handle, N, dx, incx, stride_x, dy, incy, stride_y, dc, ds, batch_count)));

            if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_3 != HIPBLAS_STATUS_SUCCESS)
                    return status_3;
                if(status_4 != HIPBLAS_STATUS_SUCCESS)
                    return status_4;
            }

            host_vector<T> rx(size_x);
            host_vector<T> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general<T>(1, N, batch_count, incx, stride_x, cx, rx, rel_error);
                near_check_general<T>(1, N, batch_count, incy, stride_y, cy, ry, rel_error);
            }
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
