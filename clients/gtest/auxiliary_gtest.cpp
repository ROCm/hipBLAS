/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_exceptions.hpp"
#include "utility.h"
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

namespace
{

    TEST(hipblas_auxiliary, statusToString)
    {
        EXPECT_EQ(0,
                  strcmp("HIPBLAS_STATUS_ALLOC_FAILED",
                         hipblasStatusToString(HIPBLAS_STATUS_ALLOC_FAILED)));
    }

    TEST(hipblas_auxiliary, badOperation)
    {
        EXPECT_EQ(testing_bad_operation(), HIPBLAS_STATUS_INVALID_ENUM);
    }

    TEST(hipblas_auxiliary, createHandle)
    {
        EXPECT_EQ(testing_handle(), HIPBLAS_STATUS_SUCCESS);
    }

} // namespace
