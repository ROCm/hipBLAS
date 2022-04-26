/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 *
 * ************************************************************************ */

#include "testing_set_get_atomics_mode.hpp"
#include "utility.h"
#include <math.h>
#include <stdexcept>
#include <vector>

using std::vector;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<bool> set_get_atomics_mode_tuple;

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS set_get_atomics_mode:
=================================================================== */

/* ============================Setup Arguments======================================= */

Arguments setup_set_get_atomics_mode_arguments(set_get_atomics_mode_tuple tup)
{
    Arguments arg;
    arg.fortran = std::get<0>(tup);
    return arg;
}

class set_get_atomics_mode_gtest : public ::TestWithParam<set_get_atomics_mode_tuple>
{
protected:
    set_get_atomics_mode_gtest() {}
    virtual ~set_get_atomics_mode_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(set_get_atomics_mode_gtest, default)
{
    Arguments       arg    = setup_set_get_atomics_mode_arguments(GetParam());
    hipblasStatus_t status = testing_set_get_atomics_mode(arg);

    EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
}

INSTANTIATE_TEST_SUITE_P(hipblas_auxiliary_small,
                         set_get_atomics_mode_gtest,
                         Combine(ValuesIn(is_fortran)));
