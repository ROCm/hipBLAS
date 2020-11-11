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
hipblasStatus_t testing_rotmg_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotmgBatchedFn
        = FORTRAN ? hipblasRotmgBatched<T, true> : hipblasRotmgBatched<T, false>;

    int batch_count = arg.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    else if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    host_vector<T> hd1[batch_count];
    host_vector<T> hd2[batch_count];
    host_vector<T> hx1[batch_count];
    host_vector<T> hy1[batch_count];
    host_vector<T> hparams[batch_count];

    device_batch_vector<T> bd1(batch_count, 1);
    device_batch_vector<T> bd2(batch_count, 1);
    device_batch_vector<T> bx1(batch_count, 1);
    device_batch_vector<T> by1(batch_count, 1);
    device_batch_vector<T> bparams(batch_count, 5);

    for(int b = 0; b < batch_count; b++)
    {
        hd1[b]     = host_vector<T>(1);
        hd2[b]     = host_vector<T>(1);
        hx1[b]     = host_vector<T>(1);
        hy1[b]     = host_vector<T>(1);
        hparams[b] = host_vector<T>(5);
    }

    host_vector<T> cd1[batch_count];
    host_vector<T> cd2[batch_count];
    host_vector<T> cx1[batch_count];
    host_vector<T> cy1[batch_count];
    host_vector<T> cparams[batch_count];

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init<T>(hd1[b], 1, 1, 1);
        hipblas_init<T>(hd2[b], 1, 1, 1);
        hipblas_init<T>(hx1[b], 1, 1, 1);
        hipblas_init<T>(hy1[b], 1, 1, 1);
        hipblas_init<T>(hparams[b], 1, 5, 1);
        cd1[b]     = hd1[b];
        cd2[b]     = hd2[b];
        cx1[b]     = hx1[b];
        cy1[b]     = hy1[b];
        cparams[b] = hparams[b];
    }

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rotmg<T>(cd1[b], cd2[b], cx1[b], cy1[b], cparams[b]);
    }

    // Test host
    {
        host_vector<T> rd1[batch_count];
        host_vector<T> rd2[batch_count];
        host_vector<T> rx1[batch_count];
        host_vector<T> ry1[batch_count];
        host_vector<T> rparams[batch_count];
        T*             rd1_in[batch_count];
        T*             rd2_in[batch_count];
        T*             rx1_in[batch_count];
        T*             ry1_in[batch_count];
        T*             rparams_in[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            rd1_in[b] = rd1[b] = hd1[b];
            rd2_in[b] = rd2[b] = hd2[b];
            rx1_in[b] = rx1[b] = hx1[b];
            ry1_in[b] = ry1[b] = hy1[b];
            rparams_in[b] = rparams[b] = hparams[b];
        }

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = (hipblasRotmgBatchedFn(
            handle, rd1_in, rd2_in, rx1_in, ry1_in, rparams_in, batch_count));

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, rd1, cd1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rd2, cd2, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rx1, cx1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, ry1, cy1, rel_error);
            near_check_general<T>(1, 5, batch_count, 1, rparams, cparams, rel_error);
        }
    }

    // Test device
    {
        device_vector<T*, 0, T> dd1(batch_count);
        device_vector<T*, 0, T> dd2(batch_count);
        device_vector<T*, 0, T> dx1(batch_count);
        device_vector<T*, 0, T> dy1(batch_count);
        device_vector<T*, 0, T> dparams(batch_count);
        device_batch_vector<T>  bd1(batch_count, 1);
        device_batch_vector<T>  bd2(batch_count, 1);
        device_batch_vector<T>  bx1(batch_count, 1);
        device_batch_vector<T>  by1(batch_count, 1);
        device_batch_vector<T>  bparams(batch_count, 5);

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(hipMemcpy(bd1[b], hd1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bd2[b], hd2[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bx1[b], hx1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(by1[b], hy1[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(
                hipMemcpy(bparams[b], hparams[b], sizeof(T) * 5, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dd1, bd1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dd2, bd2, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx1, bx1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy1, by1, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(dparams, bparams, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4 = hipblasRotmgBatchedFn(handle, dd1, dd2, dx1, dy1, dparams, batch_count);

        if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_3 != HIPBLAS_STATUS_SUCCESS)
                return status_3;
            if(status_4 != HIPBLAS_STATUS_SUCCESS)
                return status_4;
        }

        host_vector<T> rd1[batch_count];
        host_vector<T> rd2[batch_count];
        host_vector<T> rx1[batch_count];
        host_vector<T> ry1[batch_count];
        host_vector<T> rparams[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            rd1[b]     = host_vector<T>(1);
            rd2[b]     = host_vector<T>(1);
            rx1[b]     = host_vector<T>(1);
            ry1[b]     = host_vector<T>(1);
            rparams[b] = host_vector<T>(5);
            CHECK_HIP_ERROR(hipMemcpy(rd1[b], bd1[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rd2[b], bd2[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rx1[b], bx1[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry1[b], by1[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(
                hipMemcpy(rparams[b], bparams[b], sizeof(T) * 5, hipMemcpyDeviceToHost));
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, rd1, cd1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rd2, cd2, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rx1, cx1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, ry1, cy1, rel_error);
            near_check_general<T>(1, 5, batch_count, 1, rparams, cparams, rel_error);
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
