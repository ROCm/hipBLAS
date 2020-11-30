/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

hipblasStatus_t testing_set_get_atomics_mode(const Arguments& argus)
{
    bool FORTRAN                 = argus.fortran;
    auto hipblasSetAtomicsModeFn = FORTRAN ? hipblasSetAtomicsModeFortran : hipblasSetAtomicsMode;
    auto hipblasGetAtomicsModeFn = FORTRAN ? hipblasGetAtomicsModeFortran : hipblasGetAtomicsMode;

    hipblasStatus_t status     = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_set = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_get = HIPBLAS_STATUS_SUCCESS;

    hipblasAtomicsMode_t mode;
    hipblasHandle_t      handle;
    hipblasCreate(&handle);

    // Not checking default as rocBLAS defaults to allowed
    // and cuBLAS defaults to not allowed.
    // status = hipblasGetAtomicsModeFn(handle, &mode);
    // if(status != HIPBLAS_STATUS_SUCCESS)
    // {
    //     hipblasDestroy(handle);
    //     return status;
    // }

    // EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

    // Make sure set()/get() functions work
    status_set = hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_NOT_ALLOWED);
    status_get = hipblasGetAtomicsModeFn(handle, &mode);
    if(status_set != HIPBLAS_STATUS_SUCCESS || status_get != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status_set != HIPBLAS_STATUS_SUCCESS ? status_set : status_get;
    }

    EXPECT_EQ(HIPBLAS_ATOMICS_NOT_ALLOWED, mode);

    status_set = hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_ALLOWED);
    status_get = hipblasGetAtomicsModeFn(handle, &mode);
    if(status_set != HIPBLAS_STATUS_SUCCESS || status_get != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status_set != HIPBLAS_STATUS_SUCCESS ? status_set : status_get;
    }

    EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
