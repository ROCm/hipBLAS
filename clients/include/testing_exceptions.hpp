/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_common.hpp"

hipblasStatus_t testing_bad_operation()
{
    Arguments          argus;
    hipblasLocalHandle handle(argus);
    return hipblasSgemv(
        handle, hipblasOperation_t(-1), 0, 0, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0);
}

hipblasStatus_t testing_handle()
{
    // Test out hipblasCreate() and hipblasDestroy()
    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
}
