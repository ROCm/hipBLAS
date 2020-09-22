/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_set_get_vector.hpp"
#include "testing_set_get_vector_async.hpp"
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

typedef std::tuple<int, vector<int>, bool> set_get_vector_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {M};
// add/delete as a group
const int M_range[] = {600};

// vector of vector, each triple is a {incx, incy, incd};
// add/delete this list in pairs, like {1, 1, 1}
const vector<vector<int>> incx_incy_incd_range = {{1, 1, 1},
                                                  {1, 1, 3},
                                                  {1, 2, 1},
                                                  {1, 2, 2},
                                                  {1, 3, 1},
                                                  {1, 3, 3},
                                                  {3, 1, 1},
                                                  {3, 1, 3},
                                                  {3, 2, 1},
                                                  {3, 2, 2},
                                                  {3, 3, 1},
                                                  {3, 3, 3}};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS set_get_vector:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_set_get_vector_arguments(set_get_vector_tuple tup)
{

    int         M              = std::get<0>(tup);
    vector<int> incx_incy_incd = std::get<1>(tup);

    Arguments arg;

    // see the comments about vector_size_range above
    arg.M = M;

    // see the comments about matrix_size_range above
    arg.incx = incx_incy_incd[0];
    arg.incy = incx_incy_incd[1];
    arg.incd = incx_incy_incd[2];

    return arg;
}

class set_vector_get_vector_gtest : public ::TestWithParam<set_get_vector_tuple>
{
protected:
    set_vector_get_vector_gtest() {}
    virtual ~set_vector_get_vector_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// TEST_P(set_vector_get_vector_gtest, set_get_vector_float)
TEST_P(set_vector_get_vector_gtest, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_vector_arguments(GetParam());

    hipblasStatus_t status = testing_set_get_vector<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.incx <= 0 || arg.incy <= 0 || arg.incx <= 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(set_vector_get_vector_gtest, async_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_vector_arguments(GetParam());

    hipblasStatus_t status = testing_set_get_vector_async<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.incx <= 0 || arg.incy <= 0 || arg.incx <= 0)
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

INSTANTIATE_TEST_SUITE_P(rocblas_auxiliary_small,
                         set_vector_get_vector_gtest,
                         Combine(ValuesIn(M_range),
                                 ValuesIn(incx_incy_incd_range),
                                 ValuesIn(is_fortran)));
