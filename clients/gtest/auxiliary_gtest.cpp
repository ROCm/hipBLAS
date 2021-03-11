/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */
#pragma once

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

} // namespace
