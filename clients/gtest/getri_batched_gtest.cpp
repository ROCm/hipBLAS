/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getri_batched.hpp"
#include "utility.h"
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, double, int, bool> getri_batched_tuple;

const vector<vector<int>> matrix_size_range
    = {{-1, 1}, {10, 10}, {10, 20}, {500, 600}, {1024, 1024}};

const vector<double> stride_scale_range = {2.5};

const vector<int> batch_count_range = {-1, 0, 1, 2};

const vector<bool> is_fortran = {false, true};

Arguments setup_getri_batched_arguments(getri_batched_tuple tup)
{
    vector<int> matrix_size  = std::get<0>(tup);
    double      stride_scale = std::get<1>(tup);
    int         batch_count  = std::get<2>(tup);
    bool        fortran      = std::get<3>(tup);

    Arguments arg;

    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    return arg;
}

class getri_batched_gtest : public ::TestWithParam<getri_batched_tuple>
{
protected:
    getri_batched_gtest() {}
    virtual ~getri_batched_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(getri_batched_gtest, getri_batched_gtest_float)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getri_batched_arguments(GetParam());

    hipblasStatus_t status = testing_getri_batched<float, float>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
        }
    }
}

TEST_P(getri_batched_gtest, getri_batched_gtest_double)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getri_batched_arguments(GetParam());

    hipblasStatus_t status = testing_getri_batched<double, double>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
        }
    }
}

TEST_P(getri_batched_gtest, getri_batched_gtest_float_complex)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getri_batched_arguments(GetParam());

    hipblasStatus_t status = testing_getri_batched<hipblasComplex, float>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
        }
    }
}

TEST_P(getri_batched_gtest, getri_batched_gtest_double_complex)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getri_batched_arguments(GetParam());

    hipblasStatus_t status = testing_getri_batched<hipblasDoubleComplex, double>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a vector,
// ValuesIn takes each element (a vector), combines them, and feeds them to test_p
// The combinations are  { {M, N, lda, ldb}, stride_scale, batch_count }

INSTANTIATE_TEST_SUITE_P(hipblasGetriBatched,
                         getri_batched_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
