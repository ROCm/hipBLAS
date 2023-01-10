/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#if defined(GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST)
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

using namespace testing;

class ConfigurableEventListener : public TestEventListener
{
    TestEventListener* const eventListener;
    std::atomic_size_t       skipped_tests{0}; // Number of skipped tests.

public:
    bool showTestCases      = true; // Show the names of each test case.
    bool showTestNames      = true; // Show the names of each test.
    bool showSuccesses      = true; // Show each success.
    bool showInlineFailures = true; // Show each failure as it occurs.
    bool showEnvironment    = true; // Show the setup of the global environment.
    bool showInlineSkips    = true; // Show when we skip a test.

    explicit ConfigurableEventListener(TestEventListener* theEventListener)
        : eventListener(theEventListener)
    {
    }

    ~ConfigurableEventListener() override
    {
        delete eventListener;
    }

    void OnTestProgramStart(const UnitTest& unit_test) override
    {
        eventListener->OnTestProgramStart(unit_test);
    }

    void OnTestIterationStart(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationStart(unit_test, iteration);
    }

    void OnEnvironmentsSetUpStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpStart(unit_test);
    }

    void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsSetUpEnd(unit_test);
    }

    void OnTestCaseStart(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseStart(test_case);
    }

    void OnTestStart(const TestInfo& test_info) override
    {
        if(showTestNames)
            eventListener->OnTestStart(test_info);
    }

    void OnTestPartResult(const TestPartResult& result) override
    {
        // Additional skip controls used by rocBLAS. Might consider adding here

        // if(!strcmp(result.message(), LIMITED_RAM_STRING_GTEST))
        // {
        //     if(showInlineSkips)
        //         std::cout << "Skipped test due to limited RAM environment." << std::endl;
        //     ++skipped_tests;
        // }
        // else if(!strcmp(result.message(), LIMITED_MEMORY_STRING_GTEST))
        // {
        //     if(showInlineSkips)
        //         std::cout << "Skipped test due to limited GPU memory environment." << std::endl;
        //     ++skipped_tests;
        // }
        // else if(!strcmp(result.message(), TOO_MANY_DEVICES_STRING_GTEST))
        // {
        //     if(showInlineSkips)
        //         std::cout << "Skipped test due to too few GPUs." << std::endl;
        //     ++skipped_tests;
        // }
        eventListener->OnTestPartResult(result);
    }

    void OnTestEnd(const TestInfo& test_info) override
    {
        if(test_info.result()->Failed() ? showInlineFailures : showSuccesses)
            eventListener->OnTestEnd(test_info);
    }

    void OnTestCaseEnd(const TestCase& test_case) override
    {
        if(showTestCases)
            eventListener->OnTestCaseEnd(test_case);
    }

    void OnEnvironmentsTearDownStart(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownStart(unit_test);
    }

    void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) override
    {
        if(showEnvironment)
            eventListener->OnEnvironmentsTearDownEnd(unit_test);
    }

    void OnTestIterationEnd(const UnitTest& unit_test, int iteration) override
    {
        eventListener->OnTestIterationEnd(unit_test, iteration);
    }

    void OnTestProgramEnd(const UnitTest& unit_test) override
    {
        if(skipped_tests)
            std::cout << "[ SKIPPED  ] " << skipped_tests << " tests." << std::endl;
        eventListener->OnTestProgramEnd(unit_test);
    }
};

// Set the listener for Google Tests
static void hipblas_set_listener()
{
    // remove the default listener
    auto& listeners       = testing::UnitTest::GetInstance()->listeners();
    auto  default_printer = listeners.Release(listeners.default_result_printer());

    // add our listener, by default everything is on (the same as using the default listener)
    // here I am turning everything off so I only see the 3 lines for the result
    // (plus any failures at the end), like:

    // [==========] Running 149 tests from 53 test cases.
    // [==========] 149 tests from 53 test cases ran. (1 ms total)
    // [  PASSED  ] 149 tests.
    //
    auto* listener       = new ConfigurableEventListener(default_printer);
    auto* gtest_listener = getenv("GTEST_LISTENER");

    if(gtest_listener && !strcmp(gtest_listener, "NO_PASS_LINE_IN_LOG"))
    {
        listener->showTestNames      = false;
        listener->showSuccesses      = false;
        listener->showInlineFailures = false;
        listener->showInlineSkips    = false;
    }

    listeners.Append(listener);
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

    // Set Google Test listener
    hipblas_set_listener();

    int status = 0;

    if(!datafile)
        status = RUN_ALL_TESTS();
    else
    {
        // remove standard non-yaml based gtests defined with explicit code. This depends
        // on the GTEST name convention, so for now internal tests must follow the
        // pattern INSTANTIATE_TEST_SUITE_P(*, *_gtest, *) to be filtered from yaml set
        // via this GTEST_FLAG line:
        ::testing::GTEST_FLAG(filter) = ::testing::GTEST_FLAG(filter) + "-*_gtest.*";

        status = RUN_ALL_TESTS();
    }

    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
