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
hipblasStatus_t testing_rotm_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotmBatchedFn
        = FORTRAN ? hipblasRotmBatched<T, true> : hipblasRotmBatched<T, false>;

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

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

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    device_vector<T*, 0, T> dparam(batch_count);

    // Initial Data on CPU
    host_vector<T> hx[batch_count];
    host_vector<T> hy[batch_count];
    host_vector<T> hdata[batch_count]; //(4);
    host_vector<T> hparam[batch_count]; //(5);

    device_batch_vector<T> bx(batch_count, size_x);
    device_batch_vector<T> by(batch_count, size_y);
    device_batch_vector<T> bdata(batch_count, 4);
    device_batch_vector<T> bparam(batch_count, 5);

    for(int b = 0; b < batch_count; b++)
    {
        hx[b]     = host_vector<T>(size_x);
        hy[b]     = host_vector<T>(size_y);
        hdata[b]  = host_vector<T>(4);
        hparam[b] = host_vector<T>(5);
    }

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init<T>(hx[b], 1, N, incx);
        hipblas_init<T>(hy[b], 1, N, incy);
        hipblas_init<T>(hdata[b], 1, 4, 1);

        // CPU BLAS reference data
        cblas_rotmg<T>(&hdata[b][0], &hdata[b][1], &hdata[b][2], &hdata[b][3], hparam[b]);
    }

    constexpr int FLAG_COUNT        = 4;
    const T       FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};

    for(int i = 0; i < FLAG_COUNT; i++)
    {
        for(int b = 0; b < batch_count; b++)
            hparam[b][0] = FLAGS[i];

        host_vector<T> cx[batch_count];
        host_vector<T> cy[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            cx[b] = hx[b];
            cy[b] = hy[b];

            cblas_rotm<T>(N, cx[b], incx, cy[b], incy, hparam[b]);
        }

        if(arg.unit_check || arg.norm_check)
        {
            // Test device
            {
                status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
                for(int b = 0; b < batch_count; b++)
                {
                    CHECK_HIP_ERROR(
                        hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(
                        hipMemcpy(bparam[b], hparam[b], sizeof(T) * 5, hipMemcpyHostToDevice));
                }
                CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(dparam, bparam, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

                status_2
                    = (hipblasRotmBatchedFn(handle, N, dx, incx, dy, incy, dparam, batch_count));

                if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
                {
                    hipblasDestroy(handle);
                    if(status_1 != HIPBLAS_STATUS_SUCCESS)
                        return status_1;
                    if(status_2 != HIPBLAS_STATUS_SUCCESS)
                        return status_2;
                }

                host_vector<T> rx[batch_count];
                host_vector<T> ry[batch_count];
                for(int b = 0; b < batch_count; b++)
                {
                    rx[b] = host_vector<T>(size_x);
                    ry[b] = host_vector<T>(size_y);
                    CHECK_HIP_ERROR(
                        hipMemcpy(rx[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
                    CHECK_HIP_ERROR(
                        hipMemcpy(ry[b], by[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
                }

                if(arg.unit_check)
                {
                    near_check_general<T>(1, N, batch_count, incx, cx, rx, rel_error);
                    near_check_general<T>(1, N, batch_count, incy, cy, ry, rel_error);
                }
            }
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
