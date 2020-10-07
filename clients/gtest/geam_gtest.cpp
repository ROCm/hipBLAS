/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_geam.hpp"
#include "testing_geam_batched.hpp"
#include "testing_geam_strided_batched.hpp"
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

typedef std::tuple<vector<int>, vector<double>, vector<char>, double, int, bool> geam_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

// vector of vector, each vector is a {M, N, lda, ldb, ldc};
// add/delete as a group
const vector<vector<int>> matrix_size_range = {
    {-1, -1, -1, 1, 1},
    {5, 5, 5, 5, 5},
    {3, 33, 33, 34, 35},
    {10, 10, 100, 10, 10},
    {600, 500, 500, 600, 500},
    //                                      {1024, 1024, 1024, 1024, 1024}
};

// vector of vector, each pair is a {alpha, alphai, beta, betai};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_beta_range = {
    {2.0, -3.0, 0.0, 0.0},
    {3.0, 1.0, 1.0, -1.0},
    {0.0, 0.0, 2.0, -5.0},
    {0.0, 0.0, 0.0, 0.0},
};

// vector of vector, each pair is a {transA, transB};
// add/delete this list in pairs, like {'N', 'T'}
// for single/double precision, 'C'(conjTranspose) will downgraded to 'T' (transpose) internally in
// sgeam/dgeam,
// TODO: Conjugate was broken up to rocBLAS 3.5. Add conjugate tests when fixed.
const vector<vector<char>> transA_transB_range
    = {{'N', 'N'}, {'N', 'T'}}; //, {'C', 'N'}, {'T', 'C'}};

const vector<double> stride_scale_range = {1, 3};

const vector<int> batch_count_range = {1, 3, 5};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-3 GEAM:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_geam_arguments(geam_tuple tup)
{

    vector<int>    matrix_size   = std::get<0>(tup);
    vector<double> alpha_beta    = std::get<1>(tup);
    vector<char>   transA_transB = std::get<2>(tup);
    double         stride_scale  = std::get<3>(tup);
    int            batch_count   = std::get<4>(tup);
    bool           fortran       = std::get<5>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.M   = matrix_size[0];
    arg.N   = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.ldb = matrix_size[3];
    arg.ldc = matrix_size[4];

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    arg.alpha  = alpha_beta[0];
    arg.alphai = alpha_beta[1];
    arg.beta   = alpha_beta[2];
    arg.betai  = alpha_beta[3];

    arg.transA_option = transA_transB[0];
    arg.transB_option = transA_transB[1];

    arg.stride_scale = stride_scale;
    arg.batch_count  = batch_count;

    arg.fortran = fortran;

    arg.timing = 0;

    return arg;
}

class geam_gtest : public ::TestWithParam<geam_tuple>
{
protected:
    geam_gtest() {}
    virtual ~geam_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(geam_gtest, geam_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam<float>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(geam_gtest, geam_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam<hipblasDoubleComplex>(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(geam_gtest, geam_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam_batched<float>(arg);

    if(status == HIPBLAS_STATUS_NOT_SUPPORTED)
        return; // for cuda

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(geam_gtest, geam_batched_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam_batched<hipblasDoubleComplex>(arg);

    if(status == HIPBLAS_STATUS_NOT_SUPPORTED)
        return; // for cuda

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(geam_gtest, geam_strided_batched_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam_strided_batched<float>(arg);

    if(status == HIPBLAS_STATUS_NOT_SUPPORTED)
        return; // for cuda

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(geam_gtest, geam_strided_batched_gtest_double_complex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_geam_arguments(GetParam());

    hipblasStatus_t status = testing_geam_strided_batched<hipblasDoubleComplex>(arg);

    if(status == HIPBLAS_STATUS_NOT_SUPPORTED)
        return; // for cuda

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.M < 0 || arg.N < 0 || (arg.transA_option == 'N' ? arg.lda < arg.M : arg.lda < arg.K)
           || (arg.transB_option == 'N' ? arg.ldb < arg.K : arg.ldb < arg.N) || arg.ldc < arg.M
           || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// This function mainly test the scope of alpha_beta, transA_transB,.the scope of matrix_size_range
// is small

INSTANTIATE_TEST_SUITE_P(hipblasGeam_scalar_transpose,
                         geam_gtest,
                         Combine(ValuesIn(matrix_size_range),
                                 ValuesIn(alpha_beta_range),
                                 ValuesIn(transA_transB_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(is_fortran)));
