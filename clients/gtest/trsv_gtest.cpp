/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_trsv.hpp"
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

typedef std::tuple<vector<int>, vector<int>, double> trsv_tuple;

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

// vector of vector, each vector is a {M, N, lda};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {-1, 0, -1}, {11, 0, 11}, {16, 0, 16}, {32, 0, 32}, {65, 0, 65}
    //   {10, 10, 2},
    //   {600,500, 500},
    //   {1000, 1000, 1000},
    //   {2000, 2000, 2000},
    //   {4011, 4011, 4011},
    //   {8000, 8000, 8000}
};

// vector of vector, each element is an {incx, incy}
const vector<vector<int>> incx_incy_range = {
    {-2, 0}, {1, 0}, {0, 0}, {2, 0}
    //     {10, 100}
};

// vector, each entry is  {alpha};
// add/delete single values, like {2.0}
const vector<double> alpha_range = {0.0};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 trsv:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_trsv_arguments(trsv_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    vector<int> incx        = std::get<1>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.lda = matrix_size[2];

    // see the comments about matrix_size_range above
    arg.incx = incx[0];

    arg.timing = 0;

    return arg;
}

class blas2_trsv_gtest : public ::TestWithParam<trsv_tuple>
{
protected:
    blas2_trsv_gtest() {}
    virtual ~blas2_trsv_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(blas2_trsv_gtest, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trsv_arguments(GetParam());

    hipblasStatus_t status = testing_trsv<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.lda < arg.M)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incx <= 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_trsv_gtest, double)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trsv_arguments(GetParam());

    hipblasStatus_t status = testing_trsv<double>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.lda < arg.M)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incx <= 0)
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
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(hipblastrsv,
                        blas2_trsv_gtest,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(incx_incy_range),
                                ValuesIn(alpha_range)));
