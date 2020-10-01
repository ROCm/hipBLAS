/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "hipblas.hpp"
#include "hipblas_fortran.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

hipblasStatus_t testing_set_get_atomics_mode(Arguments argus)
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

    // Make sure default atomics mode is allowed
    status = hipblasGetAtomicsModeFn(handle, &mode);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

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
