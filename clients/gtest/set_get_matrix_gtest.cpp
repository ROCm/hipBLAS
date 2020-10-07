/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_set_get_matrix.hpp"
#include "testing_set_get_matrix_async.hpp"
#include "utility.h"
#include <functional>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <tuple>
#include <vector>

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<vector<int>, vector<int>, bool> set_get_matrix_tuple;

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

// small sizes

// vector of vector, each triple is a {rows, cols};
// add/delete this list in pairs, like {3, 4}

const vector<vector<int>> rows_cols_range = {{3, 3}, {3, 30}};

// vector of vector, each triple is a {lda, ldb, ldc};
// add/delete this list in pairs, like {3, 4, 3}

const vector<vector<int>> lda_ldb_ldc_range = {{3, 3, 3},
                                               {3, 3, 4},
                                               {3, 3, 5},
                                               {3, 4, 3},
                                               {3, 4, 4},
                                               {3, 4, 5},
                                               {3, 5, 3},
                                               {3, 5, 4},
                                               {3, 5, 5},
                                               {5, 3, 3},
                                               {5, 3, 4},
                                               {5, 3, 5},
                                               {5, 4, 3},
                                               {5, 4, 4},
                                               {5, 4, 5},
                                               {5, 5, 3},
                                               {5, 5, 4},
                                               {5, 5, 5}};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS auxiliary:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_set_get_matrix_arguments(set_get_matrix_tuple tup)
{

    vector<int> rows_cols   = std::get<0>(tup);
    vector<int> lda_ldb_ldc = std::get<1>(tup);

    Arguments arg;

    arg.rows = rows_cols[0];
    arg.cols = rows_cols[1];

    arg.lda = lda_ldb_ldc[0];
    arg.ldb = lda_ldb_ldc[1];
    arg.ldc = lda_ldb_ldc[2];

    return arg;
}

class set_matrix_get_matrix_gtest : public ::TestWithParam<set_get_matrix_tuple>
{
protected:
    set_matrix_get_matrix_gtest() {}
    virtual ~set_matrix_get_matrix_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(set_matrix_get_matrix_gtest, float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_matrix_arguments(GetParam());

    hipblasStatus_t status = testing_set_get_matrix<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.rows < 0 || arg.cols <= 0 || arg.lda <= 0 || arg.ldb <= 0 || arg.ldc <= 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(set_matrix_get_matrix_gtest, async_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_set_get_matrix_arguments(GetParam());

    hipblasStatus_t status = testing_set_get_matrix_async<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.rows < 0 || arg.cols <= 0 || arg.lda <= 0 || arg.ldb <= 0 || arg.ldc <= 0)
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

INSTANTIATE_TEST_SUITE_P(hipblasAuxiliary_small,
                         set_matrix_get_matrix_gtest,
                         Combine(ValuesIn(rows_cols_range),
                                 ValuesIn(lda_ldb_ldc_range),
                                 ValuesIn(is_fortran)));
