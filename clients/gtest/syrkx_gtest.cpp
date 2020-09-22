/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syr2k.hpp"
#include "testing_syr2k_batched.hpp"
#include "testing_syr2k_strided_batched.hpp"
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

typedef std::tuple<vector<int>, vector<double>, char, char, double, int, bool> syr2k_tuple;

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

// vector of vector, each vector is a {N, K, lda, ldb, ldc};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {{-1, -1, -1, -1, -1},
                                               {11, 6, 11, 11, 11},
                                               {16, 15, 16, 16, 16},
                                               {32, 12, 32, 32, 32},
                                               {65, 4, 65, 65, 65}};

// vector, each entry is  {alpha, alphai, beta, betai};
// add/delete single values, like {2.0}
const vector<vector<double>> alpha_beta_range
    = {{-0.5, 1.5, 2.0, 1.5}, {2.0, 1.0, 2.0, 1.0}, {0.0, 0.0, 0.0, 0.0}};

const vector<char> uplo_range = {
    'L',
    'U',
};

const vector<char> transA_range = {'N', 'T'}; // 'C' not supported yet.

const vector<double> stride_scale_range = {1.0, 2.5};
const vector<int>    batch_count_range  = {-1, 0, 1, 2, 10};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 syr2k:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_syr2k_arguments(syr2k_tuple tup)
{

    vector<int>    matrix_size  = std::get<0>(tup);
    vector<double> alpha_beta   = std::get<1>(tup);
    char           uplo         = std::get<2>(tup);
    char           transA       = std::get<3>(tup);
    double         stride_scale = std::get<4>(tup);
    int            batch_count  = std::get<5>(tup);
    bool           fortran      = std::get<6>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N   = matrix_size[0];
    arg.K   = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.ldb = matrix_size[3];
    arg.ldc = matrix_size[4];

    arg.alpha  = alpha_beta[0];
    arg.alphai = alpha_beta[1];
    arg.beta   = alpha_beta[2];
    arg.betai  = alpha_beta[3];

    arg.timing = 0;

    arg.uplo_option   = uplo;
    arg.transA_option = transA;

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    return arg;
}

class blas2_syr2k_gtest : public ::TestWithParam<syr2k_tuple>
{
protected:
    blas2_syr2k_gtest() {}
    virtual ~blas2_syr2k_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// syr2k
TEST_P(blas2_syr2k_gtest, syr2k_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K)))
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(blas2_syr2k_gtest, syr2k_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K)))
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// syr2k_batched
TEST_P(blas2_syr2k_gtest, syr2k_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K))
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_syr2k_gtest, syr2k_batched_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K))
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// syr2k_strided_batched
TEST_P(blas2_syr2k_gtest, syr2k_strided_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k_strided_batched<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K))
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

TEST_P(blas2_syr2k_gtest, syr2k_strided_batched_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syr2k_arguments(GetParam());

    hipblasStatus_t status = testing_syr2k_strided_batched<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && (arg.lda < arg.N || arg.ldb < arg.N))
           || (arg.transA_option != 'N' && (arg.lda < arg.K || arg.ldb < arg.K))
           || arg.batch_count < 0)
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

INSTANTIATE_TEST_SUITE_P(hipblasSyr2k,
                         blas2_syr2k_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(alpha_beta_range),
                                 ValuesIn(uplo_range),
                                 ValuesIn(transA_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
