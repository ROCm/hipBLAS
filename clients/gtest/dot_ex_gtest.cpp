/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_dot_batched_ex.hpp"
#include "testing_dot_ex.hpp"
#include "testing_dot_strided_batched_ex.hpp"
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
typedef std::tuple<int, vector<int>, double, int, vector<hipblasDatatype_t>, bool> dot_ex_tuple;

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

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 2}
// negative increments use absolute value for comparisons, so
// some combinations may not work as expected. {-1, -1} as done
// here is fine
const vector<vector<int>> incx_incy_range = {
    {1, 1},
    {-1, -1},
};

const double stride_scale_range[] = {1.0, 2.5};

const int batch_count_range[] = {-1, 0, 1, 2, 10};

#ifdef HIPBLAS_USE_HIP_DATATYPE

const vector<vector<hipblasDatatype_t>> precisions{
// Not supported in cuBLAS
#ifndef __HIP_PLATFORM_NVCC__
    {HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_32F},
    {HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F},
#endif

    // Supported in both rocBLAS and cuBLAS
    {HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_32F},
    {HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F},
    {HIP_R_64F, HIP_R_64F, HIP_R_64F, HIP_R_64F},
    {HIP_C_32F, HIP_C_32F, HIP_C_32F, HIP_C_32F},
    {HIP_C_64F, HIP_C_64F, HIP_C_64F, HIP_C_64F}};

#else

const vector<vector<hipblasDatatype_t>> precisions{
// Not supported in cuBLAS
#ifndef __HIP_PLATFORM_NVCC__
    {HIPBLAS_R_16B, HIPBLAS_R_16B, HIPBLAS_R_16B, HIPBLAS_R_32F},
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F},
#endif

    // Supported in both rocBLAS and cuBLAS
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_32F},
    {HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F},
    {HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F},
    {HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F},
    {HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F}};

#endif // HIPBLAS_USE_HIP_DATATYPE

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

class dot_ex_gtest : public ::TestWithParam<dot_ex_tuple>
{
protected:
    dot_ex_gtest() {}
    virtual ~dot_ex_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_dot_ex_arguments(dot_ex_tuple tup)
{
    Arguments   arg;
    vector<int> incx_incy                     = std::get<1>(tup);
    arg.stride_scale                          = std::get<2>(tup);
    arg.batch_count                           = std::get<3>(tup);
    vector<hipblasDatatype_t> precision_types = std::get<4>(tup);
    arg.fortran                               = std::get<5>(tup);

    arg.incx = incx_incy[0];
    arg.incy = incx_incy[1];

    arg.a_type       = precision_types[0];
    arg.b_type       = precision_types[1];
    arg.c_type       = precision_types[2];
    arg.compute_type = precision_types[3];

    arg.timing
        = 0; // disable timing data print out. Not supposed to collect performance data in gtest

    return arg;
}

// dot tests
TEST_P(dot_ex_gtest, dot_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dot_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

#ifndef __HIP_PLATFORM_NVCC__

TEST_P(dot_ex_gtest, dot_batched_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dot_batched_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(dot_ex_gtest, dot_strided_batched_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dot_strided_batched_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

// dotc tests
TEST_P(dot_ex_gtest, dotc_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dotc_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(dot_ex_gtest, dotc_batched_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dotc_batched_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

TEST_P(dot_ex_gtest, dotc_strided_batched_ex)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.
    Arguments       arg    = setup_dot_ex_arguments(GetParam());
    hipblasStatus_t status = testing_dotc_strided_batched_ex(arg);
    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.incx < 0 || arg.incy < 0 || arg.batch_count < 0)
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

#endif

// Values is for a single item; ValuesIn is for an array
// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
INSTANTIATE_TEST_SUITE_P(hipblasDotEx,
                         dot_ex_gtest,
                         Combine(ValuesIn(N_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(precisions),
                                 ValuesIn(is_fortran)));
