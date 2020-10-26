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
hipblasStatus_t testing_rotmg_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotmgStridedBatchedFn
        = FORTRAN ? hipblasRotmgStridedBatched<T, true> : hipblasRotmgStridedBatched<T, false>;

    int    batch_count  = arg.batch_count;
    double stride_scale = arg.stride_scale;
    int    stride_d1    = stride_scale;
    int    stride_d2    = stride_scale;
    int    stride_x1    = stride_scale;
    int    stride_y1    = stride_scale;
    int    stride_param = 5 * stride_scale;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

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

    size_t size_d1    = batch_count * stride_d1;
    size_t size_d2    = batch_count * stride_d2;
    size_t size_x1    = batch_count * stride_x1;
    size_t size_y1    = batch_count * stride_y1;
    size_t size_param = batch_count * stride_param;

    // Initial Data on CPU
    host_vector<T> hd1(size_d1);
    host_vector<T> hd2(size_d2);
    host_vector<T> hx1(size_x1);
    host_vector<T> hy1(size_y1);
    host_vector<T> hparams(size_param);

    srand(1);
    hipblas_init<T>(hparams, 1, 5, 1, stride_param, batch_count);
    hipblas_init<T>(hd1, 1, 1, 1, stride_d1, batch_count);
    hipblas_init<T>(hd2, 1, 1, 1, stride_d2, batch_count);
    hipblas_init<T>(hx1, 1, 1, 1, stride_x1, batch_count);
    hipblas_init<T>(hy1, 1, 1, 1, stride_y1, batch_count);

    host_vector<T> cparams = hparams;
    host_vector<T> cd1     = hd1;
    host_vector<T> cd2     = hd2;
    host_vector<T> cx1     = hx1;
    host_vector<T> cy1     = hy1;

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rotmg<T>(cd1 + b * stride_d1,
                       cd2 + b * stride_d2,
                       cx1 + b * stride_x1,
                       cy1 + b * stride_y1,
                       cparams + b * stride_param);
    }

    // Test host
    {
        host_vector<T> rd1     = hd1;
        host_vector<T> rd2     = hd2;
        host_vector<T> rx1     = hx1;
        host_vector<T> ry1     = hy1;
        host_vector<T> rparams = hparams;

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = (hipblasRotmgStridedBatchedFn(handle,
                                                 rd1,
                                                 stride_d1,
                                                 rd2,
                                                 stride_d2,
                                                 rx1,
                                                 stride_x1,
                                                 ry1,
                                                 stride_y1,
                                                 rparams,
                                                 stride_param,
                                                 batch_count));

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
            near_check_general<T>(1, 1, batch_count, 1, stride_d1, rd1, cd1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_d2, rd2, cd2, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_x1, rx1, cx1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_y1, ry1, cy1, rel_error);
            near_check_general<T>(1, 5, batch_count, 1, stride_param, rparams, cparams, rel_error);
        }
    }

    // Test device
    {
        device_vector<T> dd1(size_d1);
        device_vector<T> dd2(size_d2);
        device_vector<T> dx1(size_x1);
        device_vector<T> dy1(size_y1);
        device_vector<T> dparams(size_param);

        CHECK_HIP_ERROR(hipMemcpy(dd1, hd1, sizeof(T) * size_d1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dd2, hd2, sizeof(T) * size_d2, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dx1, hx1, sizeof(T) * size_x1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy1, hy1, sizeof(T) * size_y1, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dparams, hparams, sizeof(T) * size_param, hipMemcpyHostToDevice));

        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4 = (hipblasRotmgStridedBatchedFn(handle,
                                                 dd1,
                                                 stride_d1,
                                                 dd2,
                                                 stride_d2,
                                                 dx1,
                                                 stride_x1,
                                                 dy1,
                                                 stride_y1,
                                                 dparams,
                                                 stride_param,
                                                 batch_count));

        if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_3 != HIPBLAS_STATUS_SUCCESS)
                return status_3;
            if(status_4 != HIPBLAS_STATUS_SUCCESS)
                return status_4;
        }

        host_vector<T> rd1(size_d1);
        host_vector<T> rd2(size_d2);
        host_vector<T> rx1(size_x1);
        host_vector<T> ry1(size_y1);
        host_vector<T> rparams(size_param);

        CHECK_HIP_ERROR(hipMemcpy(rd1, dd1, sizeof(T) * size_d1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rd2, dd2, sizeof(T) * size_d2, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rx1, dx1, sizeof(T) * size_x1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(ry1, dy1, sizeof(T) * size_y1, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rparams, dparams, sizeof(T) * size_param, hipMemcpyDeviceToHost));

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_d1, rd1, cd1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_d2, rd2, cd2, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_x1, rx1, cx1, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_y1, ry1, cy1, rel_error);
            near_check_general<T>(1, 5, batch_count, 1, stride_param, rparams, cparams, rel_error);
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
