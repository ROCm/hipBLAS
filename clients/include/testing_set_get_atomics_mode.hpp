/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
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

    hipblasAtomicsMode_t mode;
    hipblasLocalHandle   handle(argus);

    // Not checking default as rocBLAS defaults to allowed
    // and cuBLAS defaults to not allowed.
    // CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    // EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

    // Make sure set()/get() functions work
    CHECK_HIPBLAS_ERROR(hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_NOT_ALLOWED));
    CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    EXPECT_EQ(HIPBLAS_ATOMICS_NOT_ALLOWED, mode);

    CHECK_HIPBLAS_ERROR(hipblasSetAtomicsModeFn(handle, HIPBLAS_ATOMICS_ALLOWED));
    CHECK_HIPBLAS_ERROR(hipblasGetAtomicsModeFn(handle, &mode));

    EXPECT_EQ(HIPBLAS_ATOMICS_ALLOWED, mode);

    return HIPBLAS_STATUS_SUCCESS;
}
