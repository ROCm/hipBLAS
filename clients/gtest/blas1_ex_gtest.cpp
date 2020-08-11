/* ************************************************************************
 * dotright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_axpy_ex.hpp"
#include "testing_dot_ex.hpp"
#include "testing_nrm2_ex.hpp"
#include "testing_rot_ex.hpp"
#include "testing_scal_ex.hpp"
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
typedef std::tuple<int, vector<double>, vector<int>, vector<hipblasDatatype_t>, bool>
    blas1_ex_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
      =================================================================== */

/*

When you see this error, do not hack this source code, hack the Makefile. It is due to compilation.

from ‘testing::internal::CartesianProductHolder3<testing::internal::ParamGenerator<int>,
testing::internal::ParamGenerator<std::vector<double> >,
testing::internal::ParamGenerator<std::vector<int> > >’

to ‘testing::internal::ParamGenerator<std::tuple<int, std::vector<double, std::allocator<double> >,
std::vector<int, std::allocator<int> > > >’

*/

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

int N_range[] = {-1, 10, 500, 1000, 7111, 10000};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
vector<vector<double>> alpha_beta_range = {{1.0, 0.0}, {2.0, -1.0}};

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 2}
// incx , incy must > 0, otherwise there is no real computation taking place,
// but throw a message, which will still be detected by gtest
vector<vector<int>> incx_incy_range = {
    {1, 1},
    {-1, -1},
};

const vector<vector<hipblasDatatype_t>> precisions{{}};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-1: scal, dot, nrm2, asum, amax, amin, axpy, copy, swap
=================================================================== */

class blas1_ex_gtest : public ::TestWithParam<blas1_ex_tuple>
{
protected:
    blas1_ex_gtest() {}
    virtual ~blas1_ex_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_blas1_ex_arguments(blas1_ex_tuple tup)
{

    int                       N               = std::get<0>(tup);
    vector<double>            alpha_beta      = std::get<1>(tup);
    vector<int>               incx_incy       = std::get<2>(tup);
    vector<hipblasDatatype_t> precision_types = std::get<3>(tup);
    bool                      fortran         = std::get<4>(tup);

    // the first element of alpha_beta_range is always alpha, and the second is always beta
    double alpha = alpha_beta[0];
    double beta  = alpha_beta[1];

    int incx = incx_incy[0];
    int incy = incx_incy[1];

    Arguments arg;
    arg.N     = N;
    arg.alpha = alpha;
    arg.beta  = beta;
    arg.incx  = incx;
    arg.incy  = incy;

    arg.fortran = fortran;

    arg.timing
        = 0; // disable timing data print out. Not supposed to collect performance data in gtest

    return arg;
}

// axpy
TEST_P(blas1_ex_gtest, axpy_float)
{
    Arguments       arg    = setup_blas1_ex_arguments(GetParam());
    hipblasStatus_t status = testing_axpy_ex(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(!arg.incx || !arg.incy)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// scal tests
TEST_P(blas1_ex_gtest, scal_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_blas1_ex_arguments(GetParam());
    hipblasStatus_t status = testing_scal_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incx < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// dot tests
TEST_P(blas1_ex_gtest, dot_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_blas1_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dot_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incx < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incy < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for cuda
        }
    }
}

// nrm2 tests
TEST_P(blas1_ex_gtest, nrm2_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_blas1_ex_arguments(GetParam());
    hipblasStatus_t status = testing_nrm2_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.incx < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// rot
TEST_P(blas1_ex_gtest, rot_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_blas1_ex_arguments(GetParam());
    hipblasStatus_t status = testing_rot_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
    }
}

// Values is for a single item; ValuesIn is for an array
// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { N, {alpha, beta}, {incx, incy} }
INSTANTIATE_TEST_CASE_P(hipblasBlas1Ex,
                        blas1_ex_gtest,
                        Combine(ValuesIn(N_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(incx_incy_range),
                                ValuesIn(precisions),
                                ValuesIn(is_fortran)));
