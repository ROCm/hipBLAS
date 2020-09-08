/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
// #include "utility.h"

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Allocating 12MBN
    putenv("ROCBLAS_DEVICE_MEMORY_SIZE=12582912");
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
