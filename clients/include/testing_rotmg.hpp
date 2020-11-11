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
hipblasStatus_t testing_rotmg(const Arguments& arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasRotmgFn = FORTRAN ? hipblasRotmg<T, true> : hipblasRotmg<T, false>;

    hipblasHandle_t handle;
    hipblasCreate(&handle);
    host_vector<T> params(9);

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

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
        status_1               = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        status_2               = (hipblasRotmgFn(
            handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));

        if(arg.unit_check)
            near_check_general(1, 9, 1, cparams.data(), hparams.data(), rel_error);
    }

    // Test device
    {
        device_vector<T> dparams(9);
        CHECK_HIP_ERROR(hipMemcpy(dparams, params, 9 * sizeof(T), hipMemcpyHostToDevice));
        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4
            = (hipblasRotmgFn(handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
        host_vector<T> hparams(9);
        CHECK_HIP_ERROR(hipMemcpy(hparams, dparams, 9 * sizeof(T), hipMemcpyDeviceToHost));

        if(arg.unit_check)
            near_check_general(1, 9, 1, cparams.data(), hparams.data(), rel_error);
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
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
