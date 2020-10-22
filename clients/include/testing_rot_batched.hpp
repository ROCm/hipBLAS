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
hipblasStatus_t testing_rot_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotBatchedFn
        = FORTRAN ? hipblasRotBatched<T, U, V, true> : hipblasRotBatched<T, U, V, false>;

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

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
    if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<T*, 0, T> dx(batch_count);
    device_vector<T*, 0, T> dy(batch_count);
    device_vector<U>        dc(1);
    device_vector<V>        ds(1);

    // Initial Data on CPU
    host_vector<T> hx[batch_count]; //(size_x);
    host_vector<T> hy[batch_count]; //(size_y);
    host_vector<U> hc(1);
    host_vector<V> hs(1);

    device_batch_vector<T> bx(batch_count, size_x);
    device_batch_vector<T> by(batch_count, size_y);

    for(int i = 0; i < batch_count; i++)
    {
        hx[i] = host_vector<T>(size_x);
        hy[i] = host_vector<T>(size_y);
    }

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init<T>(hx[b], 1, N, incx);
        hipblas_init<T>(hy[b], 1, N, incy);
    }

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);
    hipblas_init<int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<T> cx[batch_count];
    host_vector<T> cy[batch_count];
    for(int b = 0; b < batch_count; b++)
    {
        cx[b] = hx[b];
        cy[b] = hy[b];
    }
    // cblas_rotg<T, U>(cx, cy, hc, hs);
    // cx[0] = hx[0];
    // cy[0] = hy[0];
    for(int b = 0; b < batch_count; b++)
    {
        cblas_rot<T, U, V>(N, cx[b].data(), incx, cy[b].data(), incy, *hc, *hs);
    }

    if(arg.unit_check)
    {
        // Test host
        {
            status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            status_2 = ((hipblasRotBatchedFn(handle, N, dx, incx, dy, incy, hc, hs, batch_count)));

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
                CHECK_HIP_ERROR(hipMemcpy(rx[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry[b], by[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                near_check_general(1, N, batch_count, incx, cx, rx, rel_error);
                near_check_general(1, N, batch_count, incy, cy, ry, rel_error);
            }
        }

        // Test device
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(T) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(hipMemcpy(by[b], hy[b], sizeof(T) * size_y, hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(V), hipMemcpyHostToDevice));

            status_4 = ((hipblasRotBatchedFn(handle, N, dx, incx, dy, incy, dc, ds, batch_count)));

            if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_3 != HIPBLAS_STATUS_SUCCESS)
                    return status_3;
                if(status_4 != HIPBLAS_STATUS_SUCCESS)
                    return status_4;
            }

            host_vector<T> rx[batch_count];
            host_vector<T> ry[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                rx[b] = host_vector<T>(size_x);
                ry[b] = host_vector<T>(size_y);
                CHECK_HIP_ERROR(hipMemcpy(rx[b], bx[b], sizeof(T) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(hipMemcpy(ry[b], by[b], sizeof(T) * size_y, hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                near_check_general(1, N, batch_count, incx, cx, rx, rel_error);
                near_check_general(1, N, batch_count, incy, cy, ry, rel_error);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
