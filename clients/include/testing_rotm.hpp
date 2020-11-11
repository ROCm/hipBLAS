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
hipblasStatus_t testing_rotm(const Arguments& arg)
{
    bool FORTRAN       = arg.fortran;
    auto hipblasRotmFn = FORTRAN ? hipblasRotm<T, true> : hipblasRotm<T, false>;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<T> dx(size_x);
    device_vector<T> dy(size_y);
    device_vector<T> dparam(5);

    // Initial Data on CPU
    host_vector<T> hx(size_x);
    host_vector<T> hy(size_y);
    host_vector<T> hdata(4);
    host_vector<T> hparam(5);
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);
    hipblas_init<T>(hy, 1, N, incy);
    hipblas_init<T>(hdata, 1, 4, 1);

    // CPU BLAS reference data
    cblas_rotmg<T>(&hdata[0], &hdata[1], &hdata[2], &hdata[3], hparam);
    const int FLAG_COUNT        = 4;
    const T   FLAGS[FLAG_COUNT] = {-1, 0, 1, -2};
    for(int i = 0; i < FLAG_COUNT; ++i)
    {
        hparam[0]         = FLAGS[i];
        host_vector<T> cx = hx;
        host_vector<T> cy = hy;
        cblas_rotm<T>(N, cx, incx, cy, incy, hparam);

        if(arg.unit_check)
        {
            // Test host
            {
                status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                status_2 = hipblasRotmFn(handle, N, dx, incx, dy, incy, hparam);
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general(1, N, incx, cx.data(), rx.data(), rel_error);
                    near_check_general(1, N, incy, cy.data(), ry.data(), rel_error);
                }
            }

            // Test device
            {
                status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
                CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(T) * size_y, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(dparam, hparam, sizeof(T) * 5, hipMemcpyHostToDevice));
                status_4 = hipblasRotmFn(handle, N, dx, incx, dy, incy, dparam);
                host_vector<T> rx(size_x);
                host_vector<T> ry(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(T) * size_y, hipMemcpyDeviceToHost));
                if(arg.unit_check)
                {
                    near_check_general(1, N, incx, cx.data(), rx.data(), rel_error);
                    near_check_general(1, N, incy, cy.data(), ry.data(), rel_error);
                }
            }
        }

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
           || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
            if(status_3 != HIPBLAS_STATUS_SUCCESS)
                return status_3;
            if(status_4 != HIPBLAS_STATUS_SUCCESS)
                return status_4;
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
