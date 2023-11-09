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

#include <hipblas/hipblas.h>
#include <stdbool.h>

#ifdef __cplusplus
#include "cblas_interface.h"
#include "complex.hpp"
#include "hipblas_datatype2string.hpp"
#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/*!\file
 * \brief provide data initialization, timing, hipblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error)                        \
    do                                                \
    {                                                 \
        hipError_t error__ = (error);                 \
        if(error__ != hipSuccess)                     \
        {                                             \
            fprintf(stderr,                           \
                    "hip error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error__),       \
                    error__,                          \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while(0)

#ifdef __cplusplus

#ifndef CHECK_HIPBLAS_ERROR
#define EXPECT_HIPBLAS_STATUS(status, expected)      \
    do                                               \
    {                                                \
        hipblasStatus_t status__ = (status);         \
        if(status__ != expected)                     \
        {                                            \
            fprintf(stderr,                          \
                    "hipBLAS error: %s at %s:%d\n",  \
                    hipblasStatusToString(status__), \
                    __FILE__,                        \
                    __LINE__);                       \
            return (status__);                       \
        }                                            \
    } while(0)
#define CHECK_HIPBLAS_ERROR(STATUS) EXPECT_HIPBLAS_STATUS(STATUS, HIPBLAS_STATUS_SUCCESS)
#endif

#define BLAS_1_RESULT_PRINT                                \
    do                                                     \
    {                                                      \
        if(arg.timing)                                     \
        {                                                  \
            std::cout << "N, hipblas (us), ";              \
            if(arg.norm_check)                             \
            {                                              \
                std::cout << "CPU (us), error";            \
            }                                              \
            std::cout << std::endl;                        \
            std::cout << N << ',' << gpu_time_used << ','; \
            if(arg.norm_check)                             \
            {                                              \
                std::cout << cpu_time_used << ',';         \
                std::cout << hipblas_error;                \
            }                                              \
            std::cout << std::endl;                        \
        }                                                  \
    } while(0)

// Return true if value is NaN
template <typename T>
inline bool hipblas_isnan(T)
{
    return false;
}
inline bool hipblas_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool hipblas_isnan(float arg)
{
    return std::isnan(arg);
}

inline bool hipblas_isnan(hipblasHalf arg)
{
    auto half_data = static_cast<unsigned short>(arg);
    return (~(half_data)&0x7c00) == 0 && (half_data & 0x3ff) != 0;
}
inline bool hipblas_isnan(hipblasComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}
inline bool hipblas_isnan(hipblasDoubleComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

inline hipblasHalf float_to_half(float val)
{
#ifdef HIPBLAS_USE_HIP_HALF
    return __float2half(val);
#else
    uint16_t a = _cvtss_sh(val, 0);
    return a;
#endif
}

inline float bfloat16_to_float(hipblasBfloat16 bf)
{
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(bf.data) << 16};
    return u.fp32;
}

inline hipblasBfloat16 float_to_bfloat16(float f)
{
    hipblasBfloat16 rv;
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {f};
    if(~u.int32 & 0x7f800000)
    {
        u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
    }
    else if(u.int32 & 0xffff)
    {
        u.int32 |= 0x10000; // Preserve signaling NaN
    }
    rv.data = uint16_t(u.int32 >> 16);
    return rv;
}

inline float half_to_float(hipblasHalf val)
{
#ifdef HIPBLAS_USE_HIP_HALF
    return __half2float(val);
#else
    return _cvtsh_ss(val);
#endif
}

/* =============================================================================================== */
/* Absolute values                                                                                 */
template <typename T>
inline double hipblas_abs(const T& x)
{
    return x < 0 ? -x : x;
}

template <>
inline double hipblas_abs(const hipblasHalf& x)
{
    return std::abs(half_to_float(x));
}

template <>
inline double hipblas_abs(const hipblasBfloat16& x)
{
    return std::abs(bfloat16_to_float(x));
}

inline double hipblas_abs(const hipblasComplex& x)
{
    return abs(reinterpret_cast<const std::complex<float>&>(x));
}

inline double hipblas_abs(const hipblasDoubleComplex& x)
{
    return abs(reinterpret_cast<const std::complex<double>&>(x));
}

inline int hipblas_abs(const int& x)
{
    return x < 0 ? -x : x;
}

/* =============================================================================================== */
/* Complex / real helpers.                                                                         */
template <typename T>
static constexpr bool is_complex = false;

template <>
HIPBLAS_CLANG_STATIC constexpr bool is_complex<hipblasComplex> = true;

template <>
HIPBLAS_CLANG_STATIC constexpr bool is_complex<hipblasDoubleComplex> = true;

// Get base types from complex types.
template <typename T, typename = void>
struct real_t_impl
{
    using type = T;
};

template <typename T>
struct real_t_impl<T, std::enable_if_t<is_complex<T>>>
{
    using type = decltype(T{}.real());
};

template <typename T>
using real_t = typename real_t_impl<T>::type;

/* =============================================================================================== */
/* Epsilon helpers for near checks.                                                                */
template <typename>
HIPBLAS_CLANG_STATIC constexpr double hipblas_type_epsilon = 0;
template <>
HIPBLAS_CLANG_STATIC constexpr double
    hipblas_type_epsilon<float> = std::numeric_limits<float>::epsilon();
template <>
HIPBLAS_CLANG_STATIC constexpr double
    hipblas_type_epsilon<double> = std::numeric_limits<double>::epsilon();
template <>
HIPBLAS_CLANG_STATIC constexpr double
    hipblas_type_epsilon<hipblasComplex> = std::numeric_limits<float>::epsilon();
template <>
HIPBLAS_CLANG_STATIC constexpr double
    hipblas_type_epsilon<hipblasDoubleComplex> = std::numeric_limits<double>::epsilon();
template <>
HIPBLAS_CLANG_STATIC constexpr double hipblas_type_epsilon<
    hipblasHalf> = 0.0009765625; // in fp16 diff between 0x3C00 (1.0) and fp16 0x3C01
template <>
HIPBLAS_CLANG_STATIC constexpr double hipblas_type_epsilon<
    hipblasBfloat16> = 0.0078125; // in bf16 diff between 0x3F80 (1.0) and bf16 0x3F81 in double precision

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */

using hipblas_rng_t = std::mt19937;
extern hipblas_rng_t hipblas_rng, hipblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void hipblas_seedrand()
{
    hipblas_rng = hipblas_seed;
}

class hipblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(hipblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(hipblas_rng);
    }

    explicit operator signed char()
    {
        return static_cast<signed char>(std::uniform_int_distribution<int>{}(hipblas_rng));
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half (non-template hipblasHalf takes precedence over integer template above)
    explicit operator hipblasHalf()
    {
        return random_nan_data<hipblasHalf, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator hipblasBfloat16()
    {
        return random_nan_data<hipblasBfloat16, uint16_t, 7, 8>();
    }

    // Random NaN Complex
    explicit operator hipblasComplex()
    {
        return {float(*this), float(*this)};
    }

    // Random NaN Double Complex
    explicit operator hipblasDoubleComplex()
    {
        return {double(*this), double(*this)};
    }

    // // Currently not needed
    // // Random complex integers
    // explicit operator hipblasInt8Complex()
    // {
    //     return static_cast<int8_t>(
    //         std::uniform_int_distribution<unsigned short>(1, 3)(hipblas_rng));
    // }
};

/* ============================================================================================ */
/*! \brief negate a value */

// Can rename to simply "negate" after removing usage of `using namespace std;`
template <class T>
inline T hipblas_negate(T x)
{
    return -x;
}

template <>
inline hipblasHalf hipblas_negate(hipblasHalf arg)
{
    union
    {
        hipblasHalf fp;
        uint16_t    data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline hipblasBfloat16 hipblas_negate(hipblasBfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return T(rand() % 10 + 1);
};

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(hipblas_nan_rng{});
}

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [1,2,3] */
template <>
inline hipblasHalf random_generator<hipblasHalf>()
{
    return float_to_half(float((rand() % 3 + 1))); // generate an integer number in range [1,2,3]
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
template <>
inline hipblasBfloat16 random_generator<hipblasBfloat16>()
{
    return float_to_bfloat16(
        float((rand() % 3 + 1))); // generate an integer number in range [1,2,3]
}

// for hipblasComplex, generate 2 floats
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipblasComplex random_generator<hipblasComplex>()
{
    return hipblasComplex(rand() % 10 + 1, rand() % 10 + 1);
    return {float(rand() % 10 + 1), float(rand() % 10 + 1)};
}

// for hipblasDoubleComplex, generate 2 doubles
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipblasDoubleComplex random_generator<hipblasDoubleComplex>()
{
    return hipblasDoubleComplex(rand() % 10 + 1, rand() % 10 + 1);
    return {double(rand() % 10 + 1), double(rand() % 10 + 1)};
}

/*! \brief  generate a random number in range [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] */
template <typename T>
inline T random_generator_negative()
{
    // return rand()/( (T)RAND_MAX + 1);
    return -T(rand() % 10 + 1);
};

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasHalf random_generator_negative<hipblasHalf>()
{
    return float_to_half(-float((rand() % 3 + 1)));
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasBfloat16 random_generator_negative<hipblasBfloat16>()
{
    return float_to_bfloat16(-float((rand() % 3 + 1)));
};

// for complex, generate two values, convert both to negative
/*! \brief  generate a random real value in range [-1, -10] and random
*           imaginary value in range [-1, -10]
*/
template <>
inline hipblasComplex random_generator_negative<hipblasComplex>()
{
    return {float(-(rand() % 10 + 1)), float(-(rand() % 10 + 1))};
}

template <>
inline hipblasDoubleComplex random_generator_negative<hipblasDoubleComplex>()
{
    return {double(-(rand() % 10 + 1)), double(-(rand() % 10 + 1))};
}

// HPL
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(hipblas_rng);
}

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
inline hipblasBfloat16 random_hpl_generator()
{
    return hipblasBfloat16(
        float_to_bfloat16(std::uniform_real_distribution<float>(-0.5f, 0.5f)(hipblas_rng)));
}

template <>
inline hipblasHalf random_hpl_generator()
{
    return hipblasHalf(
        float_to_half(std::uniform_real_distribution<float>(-0.5f, 0.5f)(hipblas_rng)));
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void hipblas_init(
    std::vector<T>& A, int M, int N, int lda, hipblasStride stride = 0, int batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                A[i + j * lda + b * stride] = random_generator<T>();
}

template <typename T>
void hipblas_init(T* A, int M, int N, int lda, hipblasStride stride = 0, int batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                A[i + j * lda + b * stride] = random_generator<T>();
}

template <typename T>
void hipblas_init_alternating_sign(std::vector<T>& A, int M, int N, int lda)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(int i = 0; i < M; ++i)
        for(int j = 0; j < N; ++j)
            if(j % 2 ^ i % 2)
                A[i + j * lda] = random_generator<T>();
            else
                A[i + j * lda] = random_generator_negative<T>();
}

template <typename T>
void hipblas_init_alternating_sign(
    std::vector<T>& A, int M, int N, int lda, hipblasStride stride, int batch_count)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(int i_batch = 0; i_batch < batch_count; i_batch++)
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                if(j % 2 ^ i % 2)
                    A[i + j * lda + i_batch * stride] = random_generator<T>();
                else
                    A[i + j * lda + i_batch * stride] = random_generator_negative<T>();
}

// Initialize matrix so adjacent entries have alternating sign.
template <typename T>
void hipblas_init_hpl_alternating_sign(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
            {
                auto value    = random_hpl_generator<T>();
                A[i + offset] = (i ^ j) & 1 ? value : hipblas_negate(value);
            }
        }
}

template <typename T>
void hipblas_init_hpl_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    hipblas_init_hpl_alternating_sign(A.data(), M, N, lda, stride, batch_count);
}

// Initialize vector with HPL-like random values
template <typename T>
void hipblas_init_hpl(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

template <typename T>
void hipblas_init_hpl(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_hpl_generator<T>();
}

template <typename T>
inline void
    hipblas_init_cos(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(cos(i + offset));
        }
}

template <typename T>
inline void hipblas_init_cos(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    hipblas_init_cos(A.data(), M, N, lda, stride, batch_count);
}

template <typename T>
inline void
    hipblas_init_sin(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = T(sin(i + offset));
        }
}

template <typename T>
inline void hipblas_init_sin(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    hipblas_init_sin(A.data(), M, N, lda, stride, batch_count);
}

template <>
inline void hipblas_init_cos<hipblasBfloat16>(
    hipblasBfloat16* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = float_to_bfloat16(cos(i + offset));
        }
}

template <>
inline void hipblas_init_cos<hipblasBfloat16>(std::vector<hipblasBfloat16>& A,
                                              size_t                        M,
                                              size_t                        N,
                                              size_t                        lda,
                                              size_t                        stride,
                                              size_t                        batch_count)
{
    hipblas_init_cos<hipblasBfloat16>(A.data(), M, N, lda, stride, batch_count);
}

template <>
inline void hipblas_init_sin<hipblasBfloat16>(
    hipblasBfloat16* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
#pragma omp parallel for
        for(size_t j = 0; j < N; ++j)
        {
            size_t offset = j * lda + i_batch * stride;
            for(size_t i = 0; i < M; ++i)
                A[i + offset] = float_to_bfloat16(sin(i + offset));
        }
}

template <>
inline void hipblas_init_sin<hipblasBfloat16>(std::vector<hipblasBfloat16>& A,
                                              size_t                        M,
                                              size_t                        N,
                                              size_t                        lda,
                                              size_t                        stride,
                                              size_t                        batch_count)
{
    hipblas_init_sin<hipblasBfloat16>(A.data(), M, N, lda, stride, batch_count);
}

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void hipblas_init_symmetric(std::vector<T>& A, int N, int lda)
{
    for(int i = 0; i < N; ++i)
        for(int j = 0; j <= i; ++j)
        {
            auto r         = random_generator<T>();
            A[j + i * lda] = r;
            A[i + j * lda] = r;
        }
}

/*! \brief symmetric matrix initialization for strided_batched matricies: */
template <typename T>
void hipblas_init_symmetric(
    std::vector<T>& A, int N, int lda, hipblasStride strideA, int batch_count)
{
    for(int b = 0; b < batch_count; b++)
        for(int off = b * strideA, i = 0; i < N; ++i)
            for(int j = 0; j <= i; ++j)
            {
                auto r               = random_generator<T>();
                A[i + j * lda + off] = r;
                A[j + i * lda + off] = r;
            }
}

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
void hipblas_init_hermitian(std::vector<T>& A, int N, int lda)
{
    for(int i = 0; i < N; ++i)
        for(int j = 0; j <= i; ++j)
            if(i == j)
                A[j + i * lda] = random_generator<real_t<T>>();
            else
                A[j + i * lda] = A[i + j * lda] = random_generator<T>();
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void hipblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(hipblas_nan_rng());
}

template <typename T>
inline void hipblass_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(hipblas_nan_rng());
}

/* ============================================================================================= */
/*! \brief For testing purposes, to convert a regular matrix to a packed matrix.                  */
template <typename T>
inline void regular_to_packed(bool upper, const T* A, T* AP, int n)
{
    int index = 0;
    if(upper)
        for(int i = 0; i < n; i++)
            for(int j = 0; j <= i; j++)
                AP[index++] = A[j + i * n];
    else
        for(int i = 0; i < n; i++)
            for(int j = i; j < n; j++)
                AP[index++] = A[j + i * n];
}

/* ============================================================================ */
/* \brief For testing purposes, to convert a regular matrix to a banded matrix. */
template <typename T>
inline void regular_to_banded(bool upper, const T* A, int lda, T* AB, int ldab, int n, int k)
{
    // convert regular hA matrix to banded hAB matrix.
    for(int j = 0; j < n; j++)
    {
        int min1 = upper ? std::max(0, j - k) : j;
        int max1 = upper ? j : std::min(n - 1, j + k);
        int m    = upper ? k - j : -j;

        // Move bands of hA into new banded hAB format.
        for(int i = min1; i <= max1; i++)
            AB[j * ldab + (m + i)] = A[j * lda + i];

        min1 = upper ? k + 1 : std::min(k + 1, n - j);
        max1 = ldab - 1;

        // fill in bottom with random data to ensure we aren't using it.
        // for !upper, fill in bottom right triangle as well.
        for(int i = min1; i <= max1; i++)
            hipblas_init<T>(AB + j * ldab + i, 1, 1, 1);

        // for upper, fill in top left triangle with random data to ensure
        // we aren't using it.
        if(upper)
        {
            for(int i = 0; i < m; i++)
                hipblas_init<T>(AB + j * ldab + i, 1, 1, 1);
        }
    }
}

/* ============================================================================== */
/* \brief For testing purposes, zeros out elements not needed in a banded matrix. */
template <typename T>
inline void banded_matrix_setup(bool upper, T* A, int lda, int n, int k)
{
    // Make A a banded matrix with k sub/super diagonals.
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(upper && (j > k + i || i > j))
                A[j * n + i] = T(0.0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0.0);
        }
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, makes a matrix hA into a unit_diagonal matrix and               *
 *         randomly initialize the diagonal.                                                     */
template <typename T>
void make_unit_diagonal(hipblasFillMode_t uplo, T* hA, int lda, int N)
{
    if(uplo == HIPBLAS_FILL_MODE_LOWER)
    {
        for(int i = 0; i < N; i++)
        {
            T diag = hA[i + i * N];
            for(int j = 0; j <= i; j++)
                hA[i + j * lda] = hA[i + j * lda] / diag;
        }
    }
    else // rocblas_fill_upper
    {
        for(int j = 0; j < N; j++)
        {
            T diag = hA[j + j * lda];
            for(int i = 0; i <= j; i++)
                hA[i + j * lda] = hA[i + j * lda] / diag;
        }
    }

    // randomly initalize diagonal to ensure we aren't using it's values for tests.
    for(int i = 0; i < N; i++)
    {
        hipblas_init<T>(hA + i * lda + i, 1, 1, 1);
    }
}

/* ============================================================================================= */
/*! \brief For testing purposes, prepares matrix hA for a triangular solve.                      *
 *         Makes hA strictly diagonal dominant (SPD), then calculates Cholesky factorization     *
 *         of hA.                                                                                */
template <typename T>
void prepare_triangular_solve(T* hA, int lda, T* AAT, int N, char char_uplo)
{
    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T>(HIPBLAS_OP_N, HIPBLAS_OP_C, N, N, N, T(1.0), hA, lda, hA, lda, T(0.0), AAT, lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < N; i++)
    {
        T t = 0.0;
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += hipblas_abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, N, hA, lda);
}

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipblas_float_complex -> 'c', hipblas_double_complex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  turn float -> int, double -> int, hipblas_float_complex.real() -> int,
 * hipblas_double_complex.real() -> int */
template <typename T>
int type2int(T val);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_matrix(
    const std::vector<T>& CPU_result, const std::vector<T>& GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=%.8g, GPU result=%.8g\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda]),
                   double(GPU_result[j + i * lda]));
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix, valid for complex number  */
template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
void print_matrix(
    const std::vector<T>& CPU_result, const std::vector<T>& GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=(%.8g,%.8g), GPU result=(%.8g,%.8g)\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda].real()),
                   double(CPU_result[j + i * lda].imag()),
                   double(GPU_result[j + i * lda].real()),
                   double(GPU_result[j + i * lda].imag()));
}

/* ============================================================================================= */
/*! \brief For testing purposes, copy one matrix into another with different leading dimensions  */
template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions(T&     hB,
                                                   U&     hC,
                                                   int    M,
                                                   int    N,
                                                   size_t ldb,
                                                   size_t ldc,
                                                   size_t strideb     = 0,
                                                   size_t stridec     = 0,
                                                   int    batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
    {
        auto* B = hB + b * strideb;
        auto* C = hC + b * stridec;
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions_batched(
    T& hB, U& hC, int M, int N, size_t ldb, size_t ldc)
{
    int batch_count = hB.batch_count();
    for(int b = 0; b < batch_count; b++)
    {
        auto* B = hB[b];
        auto* C = hC[b];
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

/* =============================================================================================== */

/* ============================================================================================ */
// Return path of this executable
std::string hipblas_exepath();

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipblas_tempname();

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

typedef enum hipblasClientProcessor
{
    // matching enum used in hipGcnArch
    // only including supported types
    gfx803  = 803,
    gfx900  = 900,
    gfx906  = 906,
    gfx908  = 908,
    gfx90a  = 910,
    gfx940  = 940,
    gfx941  = 941,
    gfx942  = 942,
    gfx1010 = 1010,
    gfx1011 = 1011,
    gfx1012 = 1012,
    gfx1030 = 1030,
    gfx1031 = 1031,
    gfx1032 = 1032,
    gfx1034 = 1034,
    gfx1035 = 1035,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102
} hipblasClientProcessor;

/* get architecture number */
hipblasClientProcessor getArch();
int                    getArchMajor();

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipblas sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

#ifdef __cplusplus

struct Arguments;

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */
class hipblasLocalHandle
{
    hipblasHandle_t m_handle;
    void*           m_memory = nullptr;

public:
    hipblasLocalHandle();

    explicit hipblasLocalHandle(const Arguments& arg);

    ~hipblasLocalHandle();

    hipblasLocalHandle(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle(hipblasLocalHandle&&)      = delete;
    hipblasLocalHandle& operator=(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle& operator=(hipblasLocalHandle&&) = delete;

    // Allow hipblasLocalHandle to be used anywhere hipblas_handle is expected
    operator hipblasHandle_t&()
    {
        return m_handle;
    }
    operator const hipblasHandle_t&() const
    {
        return m_handle;
    }
};

#include "hipblas_arguments.hpp"

#endif // __cplusplus
