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
hipblasStatus_t testing_rotmg(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasRotmgFn = FORTRAN ? hipblasRotmg<T, true> : hipblasRotmg<T, false>;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    host_vector<T> params(9);

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // Initial data on CPU
    srand(1);
    hipblas_init<T>(params, 1, 9, 1);

    // CPU BLAS
    host_vector<T> cparams = params;
    cblas_rotmg<T>(&cparams[0], &cparams[1], &cparams[2], &cparams[3], &cparams[4]);

    // Test host
    {
        host_vector<T> hparams = params;
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasRotmgFn(
            handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));

        if(arg.unit_check)
            near_check_general(1, 9, 1, cparams.data(), hparams.data(), rel_error);
        if(arg.norm_check)
            hipblas_error_host = norm_check_general<T>('F', 1, 9, 1, cparams, hparams);
    }

    // Test device
    {
        device_vector<T> dparams(9);
        CHECK_HIP_ERROR(hipMemcpy(dparams, params, 9 * sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            hipblasRotmgFn(handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
        host_vector<T> hparams(9);
        CHECK_HIP_ERROR(hipMemcpy(hparams, dparams, 9 * sizeof(T), hipMemcpyDeviceToHost));

        if(arg.unit_check)
            near_check_general(1, 9, 1, cparams.data(), hparams.data(), rel_error);
        if(arg.norm_check)
            hipblas_error_device = norm_check_general<T>('F', 1, 9, 1, cparams, hparams);
    }

    if(arg.timing)
    {
        device_vector<T> dparams(9);
        CHECK_HIP_ERROR(hipMemcpy(dparams, params, 9 * sizeof(T), hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasRotmgFn(
                handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N>{}.log_args<T>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         rotmg_gflop_count<T>(),
                                         rotmg_gbyte_count<T>(),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
