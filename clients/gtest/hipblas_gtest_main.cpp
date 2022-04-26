/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 *
 * ************************************************************************ */

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <hipblas.h>

#include "utility.h"

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

static void print_version_info()
{
    // clang-format off
    std::cout << "hipBLAS version "
        STRINGIFY(hipblasVersionMajor) "."
        STRINGIFY(hipblasVersionMinor) "."
        STRINGIFY(hipblasVersionPatch) "."
        STRINGIFY(hipblasVersionTweak)
        << std::endl;
    // clang-format on
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    print_version_info();

    // print device info
    int device_count = query_device_property();
    if(device_count <= 0)
    {
        std::cerr << "Error: No devices found" << std::endl;
        return EXIT_FAILURE;
    }
    set_device(0); // use first device

    ::testing::InitGoogleTest(&argc, argv);

    int status = RUN_ALL_TESTS();
    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
