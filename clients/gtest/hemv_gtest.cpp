/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include "testing_hemv.hpp"
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

typedef std::tuple<vector<int>, vector<int>, vector<double>, char, bool> hemv_tuple;

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
const vector<vector<int>> matrix_size_range = {
    {-1, -1},
    //        {10, 10, 2},
    //        {600,500, 500},
    {1000, 1000},
    //        {2000, 2000, 2000},
    //        {4011, 4011, 4011},
    //        {8000, 8000, 8000},
};

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 1}
const vector<vector<int>> incx_incy_range = {
    {2, 1}, {0, -1}, {-1, -1},
    //              {10, 100},
};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_beta_range = {
    {1.0, 0.0},
    {-1.0, -1.0},
    {2.0, 1.0},
    {0.0, 1.0},
};

// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// shemv/dhemv,
const vector<char> transA_range = {
    'N',
    'T',
    'C',
};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 hemv:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 HEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_hemv_arguments(hemv_tuple tup)
{

    vector<int>    matrix_size = std::get<0>(tup);
    vector<int>    incx_incy   = std::get<1>(tup);
    vector<double> alpha_beta  = std::get<2>(tup);
    char           transA      = std::get<3>(tup);
    bool           fortran     = std::get<4>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    // see the comments about matrix_size_range above
    arg.incx = incx_incy[0];
    arg.incy = incx_incy[1];

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha = alpha_beta[0];
    arg.beta  = alpha_beta[1];

    arg.transA_option = transA;

    arg.fortran = fortran;

    arg.timing = 0;

    return arg;
}

class hemv_gtest : public ::TestWithParam<hemv_tuple>
{
protected:
    hemv_gtest() {}
    virtual ~hemv_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(hemv_gtest, hemv_gtest_float_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_hemv_arguments(GetParam());

    hipblasStatus_t status = testing_hemv<hipblasComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N || arg.lda < 1 || arg.incx == 0 || arg.incy == 0)
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
// The combinations are  { {M, N, lda}, {incx,incy} {alpha, beta}, {transA} }

INSTANTIATE_TEST_SUITE_P(hipblasHemv,
                         hemv_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(alpha_beta_range),
                                 ValuesIn(transA_range),
                                 ValuesIn(is_fortran)));
