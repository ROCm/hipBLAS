/* ************************************************************************
 * dotright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_axpy_batched_ex.hpp"
#include "testing_axpy_ex.hpp"
#include "testing_axpy_strided_batched_ex.hpp"
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
typedef std::tuple<int, double, vector<int>, double, int, vector<hipblasDatatype_t>, bool>
    axpy_ex_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
      =================================================================== */

/*

When you see this error, do not hack this source code, hack the Makefile. It is due to compilation.

from 'testing::internal::CartesianProductHolder3<testing::internal::ParamGenerator<int>,
testing::internal::ParamGenerator<std::vector<double> >,
testing::internal::ParamGenerator<std::vector<int> > >'

to 'testing::internal::ParamGenerator<std::tuple<int, std::vector<double, std::allocator<double> >,
std::vector<int, std::allocator<int> > > >'

*/

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

const int N_range[] = {-1, 10, 500, 1000, 7111};

// vector of vector, each pair is a {alpha, beta};
// add/delete this list in pairs, like {2.0, 4.0}
const double alpha_range[] = {1.0, 2.0};

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 2}
// incx , incy must > 0, otherwise there is no real computation taking place,
// but throw a message, which will still be detected by gtest
const vector<vector<int>> incx_incy_range = {
    {1, 1},
    {-1, -1},
};

const double stride_scale_range[] = {1.0, 2.5};

const int batch_count_range[] = {-1, 0, 1, 2, 10};

// Supported rocBLAS configs
const vector<vector<hipblasDatatype_t>> precisions{
    // No cuBLAS support
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F},
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_32F},

    {HIPBLAS_R_32F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_32F},
    {HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F},
    {HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F},
    {HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F},
    {HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F}};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

class axpy_ex_gtest : public ::TestWithParam<axpy_ex_tuple>
{
protected:
    axpy_ex_gtest() {}
    virtual ~axpy_ex_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_axpy_ex_arguments(axpy_ex_tuple tup)
{
    Arguments arg;

    arg.N                                     = std::get<0>(tup);
    arg.alpha                                 = std::get<1>(tup);
    arg.incx                                  = std::get<2>(tup)[0];
    arg.incy                                  = std::get<2>(tup)[1];
    arg.stride_scale                          = std::get<3>(tup);
    arg.batch_count                           = std::get<4>(tup);
    vector<hipblasDatatype_t> precision_types = std::get<5>(tup);
    arg.fortran                               = std::get<6>(tup);

    arg.a_type       = precision_types[0];
    arg.b_type       = precision_types[1];
    arg.c_type       = precision_types[2];
    arg.compute_type = precision_types[3];

    arg.timing
        = 0; // disable timing data print out. Not supposed to collect performance data in gtest

    return arg;
}

// axpy
TEST_P(axpy_ex_gtest, axpy_ex)
{
    Arguments       arg    = setup_axpy_ex_arguments(GetParam());
    hipblasStatus_t status = testing_axpy_ex(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || !arg.incx || !arg.incy)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else if(arg.a_type == HIPBLAS_R_16F)
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // unsupported CUDA configs
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(axpy_ex_gtest, axpy_batched_ex)
{
    Arguments       arg    = setup_axpy_ex_arguments(GetParam());
    hipblasStatus_t status = testing_axpy_batched_ex(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || !arg.incx || !arg.incy || arg.batch_count <= 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for CUDA
        }
    }
}

TEST_P(axpy_ex_gtest, axpy_strided_batched_ex)
{
    Arguments       arg    = setup_axpy_ex_arguments(GetParam());
    hipblasStatus_t status = testing_axpy_strided_batched_ex(arg);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || !arg.incx || !arg.incy || arg.batch_count <= 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status); // for CUDA
        }
    }
}

// Values is for a single item; ValuesIn is for an array
// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
INSTANTIATE_TEST_CASE_P(hipblasAxpyEx,
                        axpy_ex_gtest,
                        Combine(ValuesIn(N_range),
                                ValuesIn(alpha_range),
                                ValuesIn(incx_incy_range),
                                ValuesIn(stride_scale_range),
                                ValuesIn(batch_count_range),
                                ValuesIn(precisions),
                                ValuesIn(is_fortran)));
