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

#pragma once
#ifndef _HIPBLAS_ARGUMENTS_HPP_
#define _HIPBLAS_ARGUMENTS_HPP_

#include "complex.hpp"
#include "hipblas.h"
#include "hipblas_datatype2string.hpp"
#include "utility.h"
#include <cmath>
#include <immintrin.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <vector>

// Predeclare enumerator
enum hipblas_argument : int;

// conversion helpers

template <typename T>
inline T convert_alpha_beta(double r, double i)
{
    return T(r);
}

template <>
inline hipblasHalf convert_alpha_beta<hipblasHalf>(double r, double i)
{
    return float_to_half(r);
}

template <>
inline hipblasComplex convert_alpha_beta<hipblasComplex>(double r, double i)
{
    return hipblasComplex(r, i);
}

template <>
inline hipblasDoubleComplex convert_alpha_beta<hipblasDoubleComplex>(double r, double i)
{
    return hipblasDoubleComplex(r, i);
}

/*! \brief Class used to parse command arguments in both benchmark & gtest   */
struct Arguments
{
    // if you add or reorder members you must update FOR_EACH_ARGUMENT macro

    int M  = 128;
    int N  = 128;
    int K  = 128;
    int KL = 128;
    int KU = 128;

    int rows = 128;
    int cols = 128;

    int lda = 128;
    int ldb = 128;
    int ldc = 128;
    int ldd = 128;

    hipblasDatatype_t a_type       = HIPBLAS_R_32F;
    hipblasDatatype_t b_type       = HIPBLAS_R_32F;
    hipblasDatatype_t c_type       = HIPBLAS_R_32F;
    hipblasDatatype_t d_type       = HIPBLAS_R_32F;
    hipblasDatatype_t compute_type = HIPBLAS_R_32F;

    int incx = 1;
    int incy = 1;
    int incd = 1;
    int incb = 1;

    double        stride_scale = 1.0;
    hipblasStride stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    hipblasStride stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    hipblasStride stride_c; //  stride_c > ldc * N
    hipblasStride stride_d; //  stride_d > ldd * N
    hipblasStride stride_x;
    hipblasStride stride_y;

    int start = 1024;
    int end   = 10240;
    int step  = 1000;

    double alpha  = 1.0;
    double alphai = 0.0;
    double beta   = 0.0;
    double betai  = 0.0;

    char transA = 'N';
    char transB = 'N';
    char side   = 'L';
    char uplo   = 'L';
    char diag   = 'N';

    int apiCallCount = 1;
    int batch_count  = 10;

    bool fortran = false;
    bool inplace = false; // only for trmm

    int      norm_check = 0;
    int      unit_check = 1;
    int      timing     = 0;
    int      iters      = 10;
    int      cold_iters = 2;
    uint32_t algo;
    int32_t  solution_index;
    uint32_t flags;
    char     function[64];
    char     name[64];
    char     category[64];

    int atomics_mode = HIPBLAS_ATOMICS_NOT_ALLOWED;

    hipblas_initialization initialization = hipblas_initialization::rand_int;

    // clang-format off

// Generic macro which operates over the list of arguments in order of declaration
#define FOR_EACH_ARGUMENT(OPER, SEP) \
    OPER(M) SEP                      \
    OPER(N) SEP                      \
    OPER(K) SEP                      \
    OPER(KL) SEP                     \
    OPER(KU) SEP                     \
    OPER(rows) SEP                   \
    OPER(cols) SEP                   \
    OPER(lda) SEP                    \
    OPER(ldb) SEP                    \
    OPER(ldc) SEP                    \
    OPER(ldd) SEP                    \
    OPER(a_type) SEP                 \
    OPER(b_type) SEP                 \
    OPER(c_type) SEP                 \
    OPER(d_type) SEP                 \
    OPER(compute_type) SEP           \
    OPER(incx) SEP                   \
    OPER(incy) SEP                   \
    OPER(incd) SEP                   \
    OPER(incb) SEP                   \
    OPER(stride_scale) SEP           \
    OPER(stride_a) SEP               \
    OPER(stride_b) SEP               \
    OPER(stride_c) SEP               \
    OPER(stride_d) SEP               \
    OPER(stride_x) SEP               \
    OPER(stride_y) SEP               \
    OPER(start) SEP                  \
    OPER(end) SEP                    \
    OPER(step) SEP                   \
    OPER(alpha) SEP                  \
    OPER(alphai) SEP                 \
    OPER(beta) SEP                   \
    OPER(betai) SEP                  \
    OPER(transA) SEP                 \
    OPER(transB) SEP                 \
    OPER(side) SEP                   \
    OPER(uplo) SEP                   \
    OPER(diag) SEP                   \
    OPER(apiCallCount) SEP           \
    OPER(batch_count) SEP            \
    OPER(fortran) SEP                \
    OPER(inplace) SEP                \
    OPER(norm_check) SEP             \
    OPER(unit_check) SEP             \
    OPER(timing) SEP                 \
    OPER(iters) SEP                  \
    OPER(cold_iters) SEP             \
    OPER(algo) SEP                   \
    OPER(solution_index) SEP         \
    OPER(flags) SEP                  \
    OPER(function) SEP               \
    OPER(name) SEP                   \
    OPER(category) SEP               \
    OPER(atomics_mode) SEP           \
    OPER(initialization)

    // clang-format on

    // Validate input format.
    static void validate(std::istream& ifs);

    // Function to print Arguments out to stream in YAML format
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Google Tests uses this with std:ostream automatically to dump parameters
    //friend std::ostream& operator<<(std::ostream& str, const Arguments& arg);

    // Function to read Arguments data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg);

    // Convert (alpha, alphai) and (beta, betai) to a particular type
    // Return alpha, beta adjusted to 0 for when they are NaN
    template <typename T>
    T get_alpha() const
    {
        return hipblas_isnan(alpha) || (is_complex<T> && hipblas_isnan(alphai))
                   ? T(0.0)
                   : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return hipblas_isnan(beta) || (is_complex<T> && hipblas_isnan(betai))
                   ? T(0.0)
                   : convert_alpha_beta<T>(beta, betai);
    }

private:
};

// We make sure that the Arguments struct is C-compatible
/*
static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");
*/

// Arguments enumerators
// Create
//     enum hipblas_argument : int {e_M, e_N, e_K, e_KL, ... };
// There is an enum value for each case in FOR_EACH_ARGUMENT.
//
#define CREATE_ENUM(NAME) e_##NAME,
enum hipblas_argument : int
{
    FOR_EACH_ARGUMENT(CREATE_ENUM, )
};
#undef CREATE_ENUM

// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m);};
// This lambda can be used to print "M" and arg.m.
//
// alpha and beta are specialized separately, because they need to use get_alpha() or get_beta().
// To prevent multiple definitions of specializations for alpha and beta, the hipblas_argument
// enum for alpha and beta are changed to hipblas_argument(-1) and hipblas_argument(-2) during
// the FOR_EACH_ARGUMENT loop. Those out-of-range enum values are not used except here, and are
// only used so that the FOR_EACH_ARGUMENT loop can be used to loop over all of the arguments.

#if __cplusplus >= 201703L
// C++17
// ArgumentsHelper contains a templated lambda apply<> where there is a template
// specialization for each line in the CPP macro FOR_EACH_ARGUMENT. For example,
// the first lambda is:  apply<e_M> = [](auto&& func, const Arguments& arg, auto){func("M", arg.m)}
// This lambda can be used to print "M" and arg.m
namespace ArgumentsHelper
{
    template <hipblas_argument>
    static constexpr auto apply = nullptr;

    // Macro defining specializations for specific arguments
    // e_alpha and e_beta get turned into negative sentinel value specializations
    // clang-format off
#define APPLY(NAME)                                                                         \
    template <>                                                                             \
    HIPBLAS_CLANG_STATIC constexpr auto                                                     \
        apply<e_##NAME == e_alpha ? hipblas_argument(-1)                                    \
                                  : e_##NAME == e_beta ? hipblas_argument(-2) : e_##NAME> = \
            [](auto&& func, const Arguments& arg, auto) { func(#NAME, arg.NAME); }

    // Specialize apply for each Argument
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_alpha
    template <>
    HIPBLAS_CLANG_STATIC constexpr auto apply<e_alpha> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("alpha", arg.get_alpha<decltype(T)>());
        };

    // Specialization for e_beta
    template <>
    HIPBLAS_CLANG_STATIC constexpr auto apply<e_beta> =
        [](auto&& func, const Arguments& arg, auto T) {
            func("beta", arg.get_beta<decltype(T)>());
        };
};
    // clang-format on

#else

// C++14. TODO: Remove when C++17 is used
// clang-format off
namespace ArgumentsHelper
{
#define APPLY(NAME)                                             \
    template <>                                                 \
    struct apply<e_##NAME == e_alpha ? hipblas_argument(-1) :   \
                 e_##NAME == e_beta  ? hipblas_argument(-2) :   \
                 e_##NAME>                                      \
    {                                                           \
        auto operator()()                                       \
        {                                                       \
            return                                              \
                [](auto&& func, const Arguments& arg, auto)     \
                {                                               \
                    func(#NAME, arg.NAME);                      \
                };                                              \
        }                                                       \
    };

    template <hipblas_argument>
    struct apply
    {
    };

    // Go through every argument and define specializations
    FOR_EACH_ARGUMENT(APPLY, ;);

    // Specialization for e_alpha
    template <>
    struct apply<e_alpha>
    {
        auto operator()()
        {
            return
                [](auto&& func, const Arguments& arg, auto T)
                {
                    func("alpha", arg.get_alpha<decltype(T)>());
                };
        }
    };

    // Specialization for e_beta
    template <>
    struct apply<e_beta>
    {
        auto operator()()
        {
            return
                [](auto&& func, const Arguments& arg, auto T)
                {
                    func("beta", arg.get_beta<decltype(T)>());
                };
        }
    };
};
// clang-format on
#endif

#undef APPLY

#endif
