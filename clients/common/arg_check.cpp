/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <iostream>
#include "hipblas.h"
#include "arg_check.h"

void verify_hipblas_status_invalid_value(hipblasStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
#endif
    if (status != HIPBLAS_STATUS_INVALID_VALUE)
    {
        std::cout << message << std::endl;
    }
}

