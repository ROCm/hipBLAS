/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "hipblas.h"

#ifdef __cplusplus
#include "complex.hpp"
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

#if __clang__
template <>
static constexpr bool is_complex<hipblasComplex> = true;

template <>
static constexpr bool is_complex<hipblasDoubleComplex> = true;
#else
template <>
constexpr bool is_complex<hipblasComplex> = true;

template <>
constexpr bool is_complex<hipblasDoubleComplex> = true;
#endif

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

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipblas_float_complex -> 'c', hipblas_double_complex
 * -> 'z' */
template <typename T>
char type2char();

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

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

char hipblas2char_operation(hipblasOperation_t value);

char hipblas2char_fill(hipblasFillMode_t value);

char hipblas2char_diagonal(hipblasDiagType_t value);

char hipblas2char_side(hipblasSideMode_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value);

hipblasFillMode_t char2hipblas_fill(char value);

hipblasDiagType_t char2hipblas_diagonal(char value);

hipblasSideMode_t char2hipblas_side(char value);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

#ifdef __cplusplus

/*! \brief Class used to parse command arguments in both client & gtest   */
class Arguments
{
public:
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

    hipblasDatatype_t a_type       = HIPBLAS_R_32F;
    hipblasDatatype_t b_type       = HIPBLAS_R_32F;
    hipblasDatatype_t c_type       = HIPBLAS_R_32F;
    hipblasDatatype_t compute_type = HIPBLAS_R_32F;

    int incx = 1;
    int incy = 1;
    int incd = 1;

    double stride_scale = 0.0;

    int start = 1024;
    int end   = 10240;
    int step  = 1000;

    double alpha  = 1.0;
    double alphai = 0.0;
    double beta   = 0.0;
    double betai  = 0.0;

    char transA_option = 'N';
    char transB_option = 'N';
    char side_option   = 'L';
    char uplo_option   = 'L';
    char diag_option   = 'N';

    int apiCallCount = 1;
    int batch_count  = 10;

    int norm_check = 0;
    int unit_check = 1;
    int timing     = 0;

    Arguments& operator=(const Arguments& rhs)
    {
        M  = rhs.M;
        N  = rhs.N;
        K  = rhs.K;
        KL = rhs.KL;
        KU = rhs.KU;

        lda = rhs.lda;
        ldb = rhs.ldb;
        ldc = rhs.ldc;

        incx = rhs.incx;
        incy = rhs.incy;
        incd = rhs.incd;

        stride_scale = rhs.stride_scale;

        start = rhs.start;
        end   = rhs.end;
        step  = rhs.step;

        alpha  = rhs.alpha;
        alphai = rhs.alphai;
        beta   = rhs.beta;
        betai  = rhs.betai;

        transA_option = rhs.transA_option;
        transB_option = rhs.transB_option;
        side_option   = rhs.side_option;
        uplo_option   = rhs.uplo_option;
        diag_option   = rhs.diag_option;

        apiCallCount = rhs.apiCallCount;
        batch_count  = rhs.batch_count;

        norm_check = rhs.norm_check;
        unit_check = rhs.unit_check;
        timing     = rhs.timing;

        return *this;
    }

    template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
    T get_alpha()
    {
        return T(alpha, alphai);
    }

    template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
    T get_alpha()
    {
        return T(alpha);
    }

    template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
    T get_beta()
    {
        return T(beta, betai);
    }

    template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
    T get_beta()
    {
        return T(beta);
    }
};

#endif // __cplusplus
#endif
