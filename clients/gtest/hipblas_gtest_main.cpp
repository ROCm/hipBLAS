/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>
#endif
#include <stdexcept>

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
