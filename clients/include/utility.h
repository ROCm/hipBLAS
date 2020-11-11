/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "hipblas.h"

#ifdef __cplusplus
#include "cblas_interface.h"
#include "complex.hpp"
#include "hipblas_datatype2string.hpp"
#include <cmath>
#include <immintrin.h>
#include <random>
#include <type_traits>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*!\file
 * \brief provide data initialization, timing, hipblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error)                    \
    do                                            \
    {                                             \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

#ifdef __cplusplus

#define BLAS_1_RESULT_PRINT                                \
    do                                                     \
    {                                                      \
        if(argus.timing)                                   \
        {                                                  \
            std::cout << "N, hipblas (us), ";              \
            if(argus.norm_check)                           \
            {                                              \
                std::cout << "CPU (us), error";            \
            }                                              \
            std::cout << std::endl;                        \
            std::cout << N << ',' << gpu_time_used << ','; \
            if(argus.norm_check)                           \
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
    return (~arg & 0x7c00) == 0 && (arg & 0x3ff) != 0;
}
inline bool hipblas_isnan(hipblasComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}
inline bool hipblas_isnan(hipblasDoubleComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline hipblasHalf float_to_half(float val)
{
    // return static_cast<hipblasHalf>( _mm_cvtsi128_si32( _mm_cvtps_ph( _mm_set_ss( val ), 0 ) )
    uint16_t a = _cvtss_sh(val, 0);
    return a;
}

// Helper routine to convert halfs into their floats equivalent; uses F16C instructions
inline float half_to_float(hipblasHalf val)
{
    // return static_cast<hipblasHalf>(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(val), 0)));
    return _cvtsh_ss(val);
}

// zero extend lower 16 bits of bfloat16 to convert to IEEE float
inline float bfloat16_to_float(hipblasBfloat16 val)
{
    union
    {
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(val.data) << 16};
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
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return T(rand() % 10 + 1);
};

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

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief Packs strided_batched matricies into groups of 4 in N */
template <typename T>
void hipblas_packInt8(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t batch_count = 1, size_t stride_a = 0)
{
    std::vector<T> temp(A);
    for(size_t b = 0; b < batch_count; b++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * b)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * b)];
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void hipblas_init(std::vector<T>& A, int M, int N, int lda, int stride = 0, int batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
        for(int i = 0; i < M; ++i)
            for(int j = 0; j < N; ++j)
                A[i + j * lda + b * stride] = random_generator<T>();
}

template <typename T>
void hipblas_init(T* A, int M, int N, int lda, int stride = 0, int batch_count = 1)
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
    std::vector<T>& A, int M, int N, int lda, int stride, int batch_count)
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
void hipblas_init_symmetric(std::vector<T>& A, int N, int lda, int strideA, int batch_count)
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
                A[j * n + i] = T(0);
            else if(!upper && (i > k + j || j > i))
                A[j * n + i] = T(0);
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
            t += std::abs(AAT[i + j * lda]);
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

/* =============================================================================================== */

/* ============================================================================================ */
// Return path of this executable
std::string hipblas_exepath();

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

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

#include "hipblas_arguments.hpp"

#endif // __cplusplus
#endif
