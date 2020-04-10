/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrs.hpp"
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

typedef std::tuple<vector<int>, double, int> getrs_tuple;

const vector<vector<int>> matrix_size_range
    = {{-1, 1, 1}, {10, 20, 100}, {500, 600, 600}, {1024, 1024, 1024}};

const vector<double> stride_scale_range = {2.5};

const vector<int> batch_count_range = {1};

Arguments setup_getrs_arguments(getrs_tuple tup)
{
    vector<int> matrix_size  = std::get<0>(tup);
    double      stride_scale = std::get<1>(tup);
    int         batch_count  = std::get<2>(tup);

    Arguments arg;

    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];
    arg.ldb = matrix_size[2];

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    return arg;
}

class getrs_gtest : public ::TestWithParam<getrs_tuple>
{
protected:
    getrs_gtest() {}
    virtual ~getrs_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(getrs_gtest, getrs_gtest_float)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getrs_arguments(GetParam());

    hipblasStatus_t status = testing_getrs<float>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(getrs_gtest, getrs_gtest_double)
{
    // GetParam returns a tuple. The setup routine unpacks the tuple
    // and initializes arg(Arguments), which will be passed to testing routine.

    Arguments arg = setup_getrs_arguments(GetParam());

    hipblasStatus_t status = testing_getrs<double>(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.ldb < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a vector,
// ValuesIn takes each element (a vector), combines them, and feeds them to test_p
// The combinations are  { {N, lda, ldb}, stride_scale, batch_count }

INSTANTIATE_TEST_CASE_P(hipblasGetrs,
                        getrs_gtest,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(stride_scale_range),
                                ValuesIn(batch_count_range)));
