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

#include "testing_gels.hpp"
#include "utility.h"
#include <math.h>
#include <stdexcept>
#include <vector>

using std::vector;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

typedef std::tuple<vector<int>, char, bool> gels_tuple;
typedef std::tuple<bool>                    gels_bad_arg_tuple;

// {m, n, nrhs, lda, ldb}
const vector<vector<int>> matrix_size_range
    = {{-1, -1, -1, 1, 1}, {10, 10, 10, 10, 10}, {10, 10, 10, 20, 100}, {600, 500, 400, 600, 600}};

const vector<char> trans_range = {
    'N',
    'T',
};

const vector<bool> is_fortran = {false, true};

Arguments setup_gels_arguments(gels_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char        trans       = std::get<1>(tup);
    bool        fortran     = std::get<2>(tup);

    Arguments arg;

    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.K   = matrix_size[2]; // nrhs
    arg.lda = matrix_size[3];
    arg.ldb = matrix_size[4];

    arg.transA = trans;

    arg.fortran = fortran;

    return arg;
}

class gels_gtest_bad_arg : public ::TestWithParam<gels_bad_arg_tuple>
{
protected:
    gels_gtest_bad_arg() {}
    virtual ~gels_gtest_bad_arg() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class gels_gtest : public ::TestWithParam<gels_tuple>
{
protected:
    gels_gtest() {}
    virtual ~gels_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

#ifndef __HIP_PLATFORM_NVCC__

TEST_P(gels_gtest_bad_arg, gels_gtest_bad_arg_test)
{
    Arguments arg;

    EXPECT_EQ(testing_gels_bad_arg<float>(arg), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(testing_gels_bad_arg<double>(arg), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(testing_gels_bad_arg<hipblasComplex>(arg), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(testing_gels_bad_arg<hipblasDoubleComplex>(arg), HIPBLAS_STATUS_SUCCESS);
}

TEST_P(gels_gtest, gels_gtest_float)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_gels_arguments(GetParam());

    hipblasStatus_t status = testing_gels<float>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M || arg.ldb < arg.M
           || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(gels_gtest, gels_gtest_double)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_gels_arguments(GetParam());

    hipblasStatus_t status = testing_gels<double>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M || arg.ldb < arg.M
           || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(gels_gtest, gels_gtest_float_complex)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_gels_arguments(GetParam());

    hipblasStatus_t status = testing_gels<hipblasComplex>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M || arg.ldb < arg.M
           || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(gels_gtest, gels_gtest_double_complex)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_gels_arguments(GetParam());

    hipblasStatus_t status = testing_gels<hipblasDoubleComplex>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || arg.K < 0 || arg.lda < arg.M || arg.ldb < arg.M
           || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a vector,
// ValuesIn takes each element (a vector), combines them, and feeds them to test_p
// The combinations are  { {M, N, nrhs, lda, ldb}, trans, fortran }

INSTANTIATE_TEST_SUITE_P(hipblasGels,
                         gels_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(trans_range),
                                 ValuesIn(is_fortran)));

INSTANTIATE_TEST_SUITE_P(hipblasGelsBadArg, gels_gtest_bad_arg, Combine(ValuesIn(is_fortran)));

#endif
