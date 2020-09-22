/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_tbmv.hpp"
#include "testing_tbmv_batched.hpp"
#include "testing_tbmv_strided_batched.hpp"
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

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, vector<int>, double, int, bool> tbmv_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {M, K, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {-1, -1, -1}, {11, 5, 11}, {16, 8, 16}, {32, 16, 32}, {65, 64, 65}
    //   {10, 10, 2},
    //   {600,500, 500},
    //   {1000, 1000, 1000},
    //   {2000, 2000, 2000},
    //   {4011, 4011, 4011},
    //   {8000, 8000, 8000}
};

// vector of vector, each element is an {incx}
const vector<vector<int>> incx_incy_range = {
    {1}, {0}, {2}
    //     {10, 100}
};

// add/delete single values, like {2.0}
const vector<double> stride_scale_range = {1.0, 2.5};

const vector<int> batch_count_range = {-1, 0, 1, 2, 10};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 tbmv:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 TBMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_tbmv_arguments(tbmv_tuple tup)
{
    vector<int> matrix_size  = std::get<0>(tup);
    vector<int> incx         = std::get<1>(tup);
    double      stride_scale = std::get<2>(tup);
    int         batch_count  = std::get<3>(tup);
    bool        fortran      = std::get<4>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.K   = matrix_size[1];
    arg.lda = matrix_size[2];

    // see the comments about matrix_size_range above
    arg.incx = incx[0];

    arg.timing = 0;

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    return arg;
}

class blas2_tbmv_gtest : public ::TestWithParam<tbmv_tuple>
{
protected:
    blas2_tbmv_gtest() {}
    virtual ~blas2_tbmv_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(blas2_tbmv_gtest, tbmv_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_tbmv_arguments(GetParam());

    hipblasStatus_t status = testing_tbmv<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.K < 0 || arg.lda < arg.M || arg.incx == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_tbmv_gtest, tbmv_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_tbmv_arguments(GetParam());

    hipblasStatus_t status = testing_tbmv<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.K < 0 || arg.lda < arg.M || arg.incx == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_tbmv_gtest, tbmv_batched_float)
{
    Arguments arg = setup_tbmv_arguments(GetParam());

    hipblasStatus_t status = testing_tbmv_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.K < 0 || arg.lda < arg.M || arg.incx == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_tbmv_gtest, tbmv_strided_batched_float)
{
    Arguments arg = setup_tbmv_arguments(GetParam());

    hipblasStatus_t status = testing_tbmv_strided_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.K < 0 || arg.lda < arg.M || arg.incx == 0 || arg.batch_count < 0)
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
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} }

INSTANTIATE_TEST_SUITE_P(hipblastbmv,
                         blas2_tbmv_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
