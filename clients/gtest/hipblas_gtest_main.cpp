/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "program_options.hpp"

#include "argument_model.hpp"
#include "hipblas_data.hpp"
#include "hipblas_parse_data.hpp"
#include "hipblas_test.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include <hipblas.h>

#include "clients_common.hpp"
#include "utility.h"

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

#if !defined(WIN32) && defined(GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST)
#define HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(testclass);
#else
#define HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass)
#endif

#define INSTANTIATE_TEST_CATEGORY(testclass, category)                                        \
    HIPBLAS_ALLOW_UNINSTANTIATED_GTEST(testclass)                                             \
    INSTANTIATE_TEST_SUITE_P(                                                                 \
        category,                                                                             \
        testclass,                                                                            \
        testing::ValuesIn(HipBLAS_TestData::begin([](const Arguments& arg) { return true; }), \
                          HipBLAS_TestData::end()),                                           \
        testclass::PrintToStringParamName());

struct data_driven : public testing::TestWithParam<Arguments>
{
    virtual void TestBody() {}

    void operator()(const Arguments& arg)
    {
        run_bench_test(const_cast<Arguments&>(arg), 1, 0);
    }

    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            std::string name(info.param.category);

            get_test_name(info.param, name);

            // random trailer used as logged Arguments may not be unique
            char buf[256];
            sprintf(buf, "_%d", rand());
            name += buf;

            return name;
        }
    };
};

TEST_P(data_driven, yaml)
{
    return data_driven()(GetParam());
}

INSTANTIATE_TEST_CATEGORY(data_driven, _);

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

int hipblas_test_datafile()
{
    int ret = 0;
    for(Arguments arg : HipBLAS_TestData())
        ret |= run_bench_test(arg, 1, 0);
    test_cleanup::cleanup();
    return ret;
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

    bool datafile = hipblas_parse_data(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);

    int status = 0;

    if(!datafile)
        status = RUN_ALL_TESTS();
    else
    {
        // remove standard non-yaml based gtests defined with explicit code
        // this depends on Gtest code name which might change
        ::testing::GTEST_FLAG(filter) = ::testing::GTEST_FLAG(filter) + "-*_gtest.*";

        status = RUN_ALL_TESTS();
    }

    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
