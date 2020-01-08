/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "hipblas.h"
#include <gtest/gtest.h>
#include <stdexcept>
//#include "utility.h"

using namespace std;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
     BLAS set-get_pointer_mode:
=================================================================== */

TEST(hipblas_set_pointer, hipblas_get_pointer)
{
    hipblasStatus_t      status = HIPBLAS_STATUS_SUCCESS;
    hipblasPointerMode_t mode   = HIPBLAS_POINTER_MODE_DEVICE;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    status = hipblasGetPointerMode(handle, &mode);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(HIPBLAS_POINTER_MODE_DEVICE, mode);

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    status = hipblasGetPointerMode(handle, &mode);
    EXPECT_EQ(status, HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(HIPBLAS_POINTER_MODE_HOST, mode);

    hipblasDestroy(handle);
}
