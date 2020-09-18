/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "testing_trtri_strided_batched.hpp"
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

typedef std::tuple<vector<int>, char, char, int, bool> trtri_tuple;

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

// vector of vector, each vector is a {N, lda}; N > 32 will return not implemented
// add/delete as a group
const vector<vector<int>> matrix_size_range
    = {{-1, -1}, {10, 10}, {20, 160}, {21, 14}, {32, 32}, {111, 122}};

const vector<char> uplo_range = {'U', 'L'};
const vector<char> diag_range = {'N', 'U'};

// it applies on trtri_batched only
const vector<int> batch_range = {-1, 1, 100, 1000};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 TRTRI and TRTRI_Batched
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// If soe, you unpack it with extreme care for each one by like "std::get<0>" which is not intuitive
// and error-prone

Arguments setup_trtri_arguments(trtri_tuple tup)
{

    vector<int> matrix_size = std::get<0>(tup);
    char        uplo        = std::get<1>(tup);
    char        diag        = std::get<2>(tup);
    int         batch_count = std::get<3>(tup);
    bool        fortran     = std::get<4>(tup);

    Arguments arg;

    arg.N   = matrix_size[0];
    arg.lda = matrix_size[1];

    arg.uplo_option = uplo;
    arg.diag_option = diag;
    arg.batch_count = batch_count;

    arg.fortran = fortran;

    arg.timing = 0;

    return arg;
}

class trtri_gtest : public ::TestWithParam<trtri_tuple>
{
protected:
    trtri_gtest() {}
    virtual ~trtri_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(trtri_gtest, trtri_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trtri_arguments(GetParam());

    hipblasStatus_t status = testing_trtri<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.lda < arg.N)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(trtri_gtest, trtri_batched_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trtri_arguments(GetParam());

    hipblasStatus_t status = testing_trtri_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {

        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(trtri_gtest, trtri_strided_batched_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_trtri_arguments(GetParam());

    hipblasStatus_t status = testing_trtri_strided_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {

        if(arg.N < 0 || arg.lda < arg.N || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// notice we are using vector of vector for matrix size, and vector for uplo, diag
// ValuesIn take each element (a vector or a char) and combine them and feed them to test_p
// The combinations are  { {N, lda}, uplo, diag }

// THis function mainly test the scope of matrix_size.
INSTANTIATE_TEST_SUITE_P(hipblasTrtri,
                         trtri_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(diag_range),
                                 ValuesIn(batch_range),
                                 ValuesIn(is_fortran)));
