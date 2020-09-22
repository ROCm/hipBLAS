/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syr2.hpp"
#include "testing_syr2_batched.hpp"
#include "testing_syr2_strided_batched.hpp"
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

typedef std::tuple<vector<int>, vector<int>, double, double, int, bool> syr2_tuple;

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

// vector of vector, each vector is a {N, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {{-1, -1}, {11, 11}, {16, 16}, {32, 32}, {65, 65}};

// vector of vector, each element is an {incx, incy}
const vector<vector<int>> incx_incy_range = {
    {-2, -2}, {1, 1}, {0, 0}, {2, 2}
    //     {10, 100}
};

// vector, each entry is  {alpha};
// add/delete single values, like {2.0}
const vector<double> alpha_range = {-0.5, 2.0, 0.0};

const vector<double> stride_scale_range = {1.0, 2.5};
const vector<int>    batch_count_range  = {-1, 0, 1, 2, 10};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 syr2:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_syr2_arguments(syr2_tuple tup)
{
    vector<int> matrix_size  = std::get<0>(tup);
    vector<int> incx_incy    = std::get<1>(tup);
    double      alpha        = std::get<2>(tup);
    double      stride_scale = std::get<3>(tup);
    int         batch_count  = std::get<4>(tup);
    bool        fortran      = std::get<5>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    // see the comments about matrix_size_range above
    arg.incx = incx_incy[0];
    arg.incy = incx_incy[0];

    arg.alpha = alpha;

    arg.timing = 0;

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    return arg;
}

class blas2_syr2_gtest : public ::TestWithParam<syr2_tuple>
{
protected:
    blas2_syr2_gtest() {}
    virtual ~blas2_syr2_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// syr2
TEST_P(blas2_syr2_gtest, syr2_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_syr2_gtest, syr2_gtest_float_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// syr2_batched
TEST_P(blas2_syr2_gtest, syr2_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_syr2_gtest, syr2_batched_gtest_float_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// syr2_strided_batched
TEST_P(blas2_syr2_gtest, syr2_strided_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2_strided_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_syr2_gtest, syr2_strided_batched_gtest_float_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2_arguments(GetParam());

    hipblasStatus_t status = testing_syr2_strided_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
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
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_SUITE_P(hipblasSyr2,
                         blas2_syr2_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(alpha_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
