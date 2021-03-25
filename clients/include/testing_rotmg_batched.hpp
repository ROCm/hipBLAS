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

template <typename T>
hipblasStatus_t testing_rotmg_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotmgBatchedFn
        = FORTRAN ? hipblasRotmgBatched<T, true> : hipblasRotmgBatched<T, false>;

    int batch_count = arg.batch_count;

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

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    host_batch_vector<T> hd1(1, 1, batch_count);
    host_batch_vector<T> hd2(1, 1, batch_count);
    host_batch_vector<T> hx1(1, 1, batch_count);
    host_batch_vector<T> hy1(1, 1, batch_count);
    host_batch_vector<T> hparams(5, 1, batch_count);
    host_batch_vector<T> cd1(1, 1, batch_count);
    host_batch_vector<T> cd2(1, 1, batch_count);
    host_batch_vector<T> cx1(1, 1, batch_count);
    host_batch_vector<T> cy1(1, 1, batch_count);
    host_batch_vector<T> cparams(5, 1, batch_count);

    device_batch_vector<T> dd1(1, 1, batch_count);
    device_batch_vector<T> dd2(1, 1, batch_count);
    device_batch_vector<T> dx1(1, 1, batch_count);
    device_batch_vector<T> dy1(1, 1, batch_count);
    device_batch_vector<T> dparams(5, 1, batch_count);

    device_batch_vector<T> bd1(batch_count, 1);
    device_batch_vector<T> bd2(batch_count, 1);
    device_batch_vector<T> bx1(batch_count, 1);
    device_batch_vector<T> by1(batch_count, 1);
    device_batch_vector<T> bparams(batch_count, 5);

    hipblas_init(hd1, true);
    hipblas_init(hd2, false);
    hipblas_init(hx1, false);
    hipblas_init(hy1, false);
    hipblas_init(hparams, false);
    cd1.copy_from(hd1);
    cd2.copy_from(hd2);
    cx1.copy_from(hx1);
    cy1.copy_from(hy1);
    cparams.copy_from(hparams);

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rotmg<T>(cd1[b], cd2[b], cx1[b], cy1[b], cparams[b]);
    }

    // Test host
    {
        host_batch_vector<T> rd1(1, 1, batch_count);
        host_batch_vector<T> rd2(1, 1, batch_count);
        host_batch_vector<T> rx1(1, 1, batch_count);
        host_batch_vector<T> ry1(1, 1, batch_count);
        host_batch_vector<T> rparams(5, 1, batch_count);
        rd1.copy_from(hd1);
        rd2.copy_from(hd2);
        rx1.copy_from(hx1);
        ry1.copy_from(hy1);
        rparams.copy_from(hparams);

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            hipblasRotmgBatchedFn(handle, rd1, rd2, rx1, ry1, rparams, batch_count));

        if(arg.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                near_check_general<T>(1, 1, 1, cd1[b], rd1[b], rel_error);
                near_check_general<T>(1, 1, 1, cd2[b], rd2[b], rel_error);
                near_check_general<T>(1, 1, 1, cx1[b], rx1[b], rel_error);
                near_check_general<T>(1, 1, 1, cy1[b], ry1[b], rel_error);
                near_check_general<T>(1, 5, 1, cparams[b], rparams[b], rel_error);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, cd1, rd1, batch_count);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cd2, rd2, batch_count);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cx1, rx1, batch_count);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cy1, ry1, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 5, 1, cparams, rparams, batch_count);
        }
    }

    // Test device
    {
        device_batch_vector<T> dd1(1, 1, batch_count);
        device_batch_vector<T> dd2(1, 1, batch_count);
        device_batch_vector<T> dx1(1, 1, batch_count);
        device_batch_vector<T> dy1(1, 1, batch_count);
        device_batch_vector<T> dparams(5, 1, batch_count);
        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx1.transfer_from(hx1));
        CHECK_HIP_ERROR(dy1.transfer_from(hy1));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasRotmgBatchedFn(handle,
                                                  dd1.ptr_on_device(),
                                                  dd2.ptr_on_device(),
                                                  dx1.ptr_on_device(),
                                                  dy1.ptr_on_device(),
                                                  dparams.ptr_on_device(),
                                                  batch_count));

        host_batch_vector<T> rd1(1, 1, batch_count);
        host_batch_vector<T> rd2(1, 1, batch_count);
        host_batch_vector<T> rx1(1, 1, batch_count);
        host_batch_vector<T> ry1(1, 1, batch_count);
        host_batch_vector<T> rparams(5, 1, batch_count);
        CHECK_HIP_ERROR(rd1.transfer_from(dd1));
        CHECK_HIP_ERROR(rd2.transfer_from(dd2));
        CHECK_HIP_ERROR(rx1.transfer_from(dx1));
        CHECK_HIP_ERROR(ry1.transfer_from(dy1));
        CHECK_HIP_ERROR(rparams.transfer_from(dparams));

        if(arg.unit_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                near_check_general<T>(1, 1, 1, cd1[b], rd1[b], rel_error);
                near_check_general<T>(1, 1, 1, cd2[b], rd2[b], rel_error);
                near_check_general<T>(1, 1, 1, cx1[b], rx1[b], rel_error);
                near_check_general<T>(1, 1, 1, cy1[b], ry1[b], rel_error);
                near_check_general<T>(1, 5, 1, cparams[b], rparams[b], rel_error);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_device = norm_check_general<T>('F', 1, 1, 1, cd1, rd1, batch_count);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cd2, rd2, batch_count);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cx1, rx1, batch_count);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cy1, ry1, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 5, 1, cparams, rparams, batch_count);
        }
    }

    if(arg.timing)
    {
        device_batch_vector<T> dd1(1, 1, batch_count);
        device_batch_vector<T> dd2(1, 1, batch_count);
        device_batch_vector<T> dx1(1, 1, batch_count);
        device_batch_vector<T> dy1(1, 1, batch_count);
        device_batch_vector<T> dparams(5, 1, batch_count);
        CHECK_HIP_ERROR(dd1.transfer_from(hd1));
        CHECK_HIP_ERROR(dd2.transfer_from(hd2));
        CHECK_HIP_ERROR(dx1.transfer_from(hx1));
        CHECK_HIP_ERROR(dy1.transfer_from(hy1));
        CHECK_HIP_ERROR(dparams.transfer_from(hparams));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasRotmgBatchedFn(handle,
                                                      dd1.ptr_on_device(),
                                                      dd2.ptr_on_device(),
                                                      dx1.ptr_on_device(),
                                                      dy1.ptr_on_device(),
                                                      dparams.ptr_on_device(),
                                                      batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_batch_count>{}.log_args<T>(std::cout,
                                                   arg,
                                                   gpu_time_used,
                                                   ArgumentLogging::NA_value,
                                                   ArgumentLogging::NA_value,
                                                   hipblas_error_host,
                                                   hipblas_error_device);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
