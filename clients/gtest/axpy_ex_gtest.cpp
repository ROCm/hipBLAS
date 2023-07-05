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

#include "testing_axpy_batched_ex.hpp"
#include "testing_axpy_ex.hpp"
#include "testing_axpy_strided_batched_ex.hpp"
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
typedef std::tuple<int, vector<double>, vector<int>, double, int, vector<hipblasDatatype_t>, bool>
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

// vector of vector, each pair is a {alpha, alphai};
// add/delete this list in pairs, like {2.0, 4.0}
const vector<vector<double>> alpha_range = {{1.0, 2.0}};

// vector of vector, each pair is a {incx, incy};
// add/delete this list in pairs, like {1, 2}
// negative increments use absolute value for comparisons, so
// some combinations may not work as expected. {-1, -1} as done
// here is fine
const vector<vector<int>> incx_incy_range = {
    {1, 1},
    {2, 3},
    {-1, -1},
};

const double stride_scale_range[] = {1.0, 2.5};

const int batch_count_range[] = {-1, 0, 1, 2, 10};

// Supported rocBLAS configs
const vector<vector<hipblasDatatype_t>> precisions{
// No cuBLAS support
#ifndef __HIP_PLATFORM_NVCC__
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F},
    {HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_32F},
#endif

    {HIPBLAS_R_32F, HIPBLAS_R_16F, HIPBLAS_R_16F, HIPBLAS_R_32F},
    {HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F, HIPBLAS_R_32F},
    {HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F, HIPBLAS_R_64F},
    {HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F, HIPBLAS_C_32F},
    {HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F, HIPBLAS_C_64F}};

// Fortran interface doesn't change when compiling with HIPBLAS_V2 and will continue to accept hipblasDatatype_t for now.
// When we remove hipblasDatatype_t, the Fortran interface will change accordingly.
// So not testing fortran interface with hipblas_v2-test.
#ifdef HIPBLAS_V2
const bool is_fortran[] = {false};
#else
const bool is_fortran[] = {false, true};
#endif

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
    arg.alpha                                 = std::get<1>(tup)[0];
    arg.alphai                                = std::get<1>(tup)[1];
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

#ifndef __HIP_PLATFORM_NVCC__

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
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
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
            EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status); // fail
        }
    }
}

#endif

// Values is for a single item; ValuesIn is for an array
// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
INSTANTIATE_TEST_SUITE_P(hipblasAxpyEx,
                         axpy_ex_gtest,
                         Combine(ValuesIn(N_range),
                                 ValuesIn(alpha_range),
                                 ValuesIn(incx_incy_range),
                                 ValuesIn(stride_scale_range),
                                 ValuesIn(batch_count_range),
                                 ValuesIn(precisions),
                                 ValuesIn(is_fortran)));
