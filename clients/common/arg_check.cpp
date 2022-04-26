/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 * ************************************************************************ */

#include "arg_check.h"
#include "hipblas.h"
#include <iostream>

void verify_hipblas_status_invalid_value(hipblasStatus_t status, const char* message)
{
#ifdef GOOGLE_TEST
    ASSERT_EQ(status, HIPBLAS_STATUS_INVALID_VALUE);
#endif
    if(status != HIPBLAS_STATUS_INVALID_VALUE)
    {
        std::cout << message << std::endl;
    }
}
