/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_hpr2.hpp"
#include "testing_hpr2_batched.hpp"
#include "testing_hpr2_strided_batched.hpp"
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

typedef std::tuple<int, vector<int>, vector<double>, char, double, int, bool> hpr2_tuple;

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

// vector of vector, each vector is a {N};
// add/delete as a group
const vector<int> matrix_size_range = {-1, 11, 16, 32, 65};

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 1}
const vector<vector<int>> incx_incy_range = {{1, 1}, {0, 0}, {2, 2}};

// vector, each entry is  {alpha};
// add/delete single values, like {2.0}
const vector<vector<double>> alpha_range = {{-0.5, 1.5}, {2.0, -1.0}, {0.0, 0.0}};

const vector<char> uplo_range = {
    'L',
    'U',
};

const vector<double> stride_scale_range = {1.0, 2.5};
const vector<int>    batch_count_range  = {-1, 0, 1, 2, 10};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 hpr2:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_hpr2_arguments(hpr2_tuple tup)
{

    int            matrix_size  = std::get<0>(tup);
    vector<int>    incx_incy    = std::get<1>(tup);
    vector<double> alpha        = std::get<2>(tup);
    char           uplo         = std::get<3>(tup);
    double         stride_scale = std::get<4>(tup);
    int            batch_count  = std::get<5>(tup);
    bool           fortran      = std::get<6>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N = matrix_size;

    // see the comments about matrix_size_range above
    arg.incx = incx_incy[0];
    arg.incy = incx_incy[1];

    arg.alpha  = alpha[0];
    arg.alphai = alpha[1];

    arg.timing = 0;

    arg.uplo_option = uplo;

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    return arg;
}

class blas2_hpr2_gtest : public ::TestWithParam<hpr2_tuple>
{
protected:
    blas2_hpr2_gtest() {}
    virtual ~blas2_hpr2_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// hpr2
TEST_P(blas2_hpr2_gtest, hpr2_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_hpr2_gtest, hpr2_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// hpr2_batched
TEST_P(blas2_hpr2_gtest, hpr2_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_hpr2_gtest, hpr2_batched_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// hpr2_strided_batched
TEST_P(blas2_hpr2_gtest, hpr2_strided_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2_strided_batched<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_hpr2_gtest, hpr2_strided_batched_gtest_double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hpr2_arguments(GetParam());

    hipblasStatus_t status = testing_hpr2_strided_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx == 0 || arg.incy == 0 || arg.batch_count < 0)
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

INSTANTIATE_TEST_SUITE_P(hipblasHpr2,
                         blas2_hpr2_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(alpha_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
