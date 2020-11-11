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

template <typename T>
hipblasStatus_t testing_rotm_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotmStridedBatchedFn
        = FORTRAN ? hipblasRotmStridedBatched<T, true> : hipblasRotmStridedBatched<T, false>;

    double stride_scale = arg.stride_scale;

    int N            = arg.N;
    int incx         = arg.incx;
    int incy         = arg.incy;
    int stride_x     = N * incx * stride_scale;
    int stride_y     = N * incy * stride_scale;
    int stride_param = 5 * stride_scale;
    int batch_count  = arg.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count <= 0)
    {
        return (batch_count < 0) ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_x     = N * size_t(incx) + size_t(stride_x) * size_t(batch_count - 1);
    size_t size_y     = N * size_t(incy) + size_t(stride_y) * size_t(batch_count - 1);
    size_t size_param = 5 + size_t(stride_param) * size_t(batch_count - 1);

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> dparam(size_param);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hdata(4 * batch_count);
    host_vector<T> hparam(size_param);
    srand(1);
    hipblas_init<T>(hx, 1, N, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, N, incy, stride_y, batch_count);
    hipblas_init<T>(hdata, 1, 4, 1, 4, batch_count);

    // CPU BLAS reference data
    for(int b = 0; b < batch_count; b++)
        cblas_rotmg<T>(hdata + b * 4,
                       hdata + b * 4 + 1,
                       hdata + b * 4 + 2,
                       hdata + b * 4 + 3,
                       hparam + b * stride_param);

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        for(int b = 0; b < batch_count; b++)
            (hparam + b * stride_param)[0] = FLAGS[i];

        host_vector<T> cx = hx;
        host_vector<T> cy = hy;

        for(int b = 0; b < batch_count; b++)
        {
            cblas_rotm<T>(
                N, cx + b * stride_x, incx, cy + b * stride_y, incy, hparam + b * stride_param);
        }

        if(arg.unit_check || arg.norm_check)
        {
            // Test device
            {
                status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(dparam, hparam, sizeof(T) * size_param, hipMemcpyHostToDevice));
                status_2 = ((hipblasRotmStridedBatchedFn(handle,
                                                         N,
                                                         dx,
                                                         incx,
                                                         stride_x,
                                                         dy,
                                                         incy,
                                                         stride_y,
                                                         dparam,
                                                         stride_param,
                                                         batch_count)));

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
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
