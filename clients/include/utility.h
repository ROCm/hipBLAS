/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_UTILITY_H_
#define _TESTING_UTILITY_H_

#include "hipblas.h"
#include <cmath>
#include <immintrin.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <typeinfo>
#include <vector>

using namespace std;

/*!\file
 * \brief provide data initialization, timing, hipblas type <-> lapack char conversion utilities.
 */

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#define BLAS_1_RESULT_PRINT                       \
    if(argus.timing)                              \
    {                                             \
        cout << "N, hipblas (us), ";              \
        if(argus.norm_check)                      \
        {                                         \
            cout << "CPU (us), error";            \
        }                                         \
        cout << endl;                             \
        cout << N << ',' << gpu_time_used << ','; \
        if(argus.norm_check)                      \
        {                                         \
            cout << cpu_time_used << ',';         \
            cout << hipblas_error;                \
        }                                         \
        cout << endl;                             \
    }

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
inline bool hipblas_isnan(hipComplex arg)
{
    return std::isnan(arg.x) || std::isnan(arg.y);
}
inline bool hipblas_isnan(hipDoubleComplex arg)
{
    return std::isnan(arg.x) || std::isnan(arg.y);
}

// Helper routine to convert floats into their half equivalent; uses F16C instructions
inline hipblasHalf float_to_half(float val)
{
    // return static_cast<hipblasHalf>( _mm_cvtsi128_si32( _mm_cvtps_ph( _mm_set_ss( val ), 0 ) )
    // );
    const int          zero = 0;
    short unsigned int a;
    a = _cvtss_sh(val, zero);
    //  return _cvtss_sh(val, zero);
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
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return (T)(rand() % 10 + 1);
};

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [1,2,3] */
template <>
inline hipblasHalf random_generator<hipblasHalf>()
{
    return float_to_half(
        static_cast<float>((rand() % 3 + 1))); // generate an integer number in range [1,2,3]
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
template <>
inline hipblasBfloat16 random_generator<hipblasBfloat16>()
{
    return float_to_bfloat16(
        static_cast<float>((rand() % 3 + 1))); // generate an integer number in range [1,2,3]
}

// for hipComplex, generate 2 floats
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipComplex random_generator<hipComplex>()
{
    return hipComplex(rand() % 10 + 1, rand() % 10 + 1);
}

// for hipDoubleComplex, generate 2 doubles
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipDoubleComplex random_generator<hipDoubleComplex>()
{
    return hipDoubleComplex(rand() % 10 + 1, rand() % 10 + 1);
}

/*! \brief  generate a random number in range [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] */
template <typename T>
T random_generator_negative()
{
    // return rand()/( (T)RAND_MAX + 1);
    return -(T)(rand() % 10 + 1);
};

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasHalf random_generator_negative<hipblasHalf>()
{
    return float_to_half(-static_cast<float>((rand() % 3 + 1)));
};

// for hipblasBfloat16, generate float, and convert to hipblasBfloat16
/*! \brief  generate a random number in range [-1,-2,-3] */
template <>
inline hipblasBfloat16 random_generator_negative<hipblasBfloat16>()
{
    return float_to_bfloat16(-static_cast<float>((rand() % 3 + 1)));
};

// for complex, generate two values, convert both to negative
/*! \brief  generate a random real value in range [-1, -10] and random
*           imaginary value in range [-1, -10]
*/
template <>
inline hipComplex random_generator_negative<hipComplex>()
{
    hipComplex(-(rand() % 10 + 1), -(rand() % 10 + 1));
}

template <>
inline hipDoubleComplex random_generator_negative<hipDoubleComplex>()
{
    hipDoubleComplex(-(rand() % 10 + 1), -(rand() % 10 + 1));
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void hipblas_init(vector<T>& A, int M, int N, int lda, int stride = 0, int batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
    {
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                A[i + j * lda + b * stride] = random_generator<T>();
            }
        }
    }
};

template <typename T>
void hipblas_init(T* A, int M, int N, int lda, int stride = 0, int batch_count = 1)
{
    for(int b = 0; b < batch_count; b++)
    {
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                A[i + j * lda + b * stride] = random_generator<T>();
            }
        }
    }
};

template <typename T>
void hipblas_init_alternating_sign(vector<T>& A, int M, int N, int lda)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(j % 2 ^ i % 2)
            {
                A[i + j * lda] = random_generator<T>();
            }
            else
            {
                A[i + j * lda] = random_generator_negative<T>();
            }
        }
    }
};

template <typename T>
void hipblas_init_alternating_sign(vector<T>& A, int M, int N, int lda, int stride, int batch_count)
{
    // Initialize matrix so adjacent entries have alternating sign.
    // In gemm if either A or B are initialized with alernating
    // sign the reduction sum will be summing positive
    // and negative numbers, so it should not get too large.
    // This helps reduce floating point inaccuracies for 16bit
    // arithmetic where the exponent has only 5 bits, and the
    // mantissa 10 bits.
    for(int i_batch = 0; i_batch < batch_count; i_batch++)
    {
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                if(j % 2 ^ i % 2)
                {
                    A[i + j * lda + i_batch * stride] = random_generator<T>();
                }
                else
                {
                    A[i + j * lda + i_batch * stride] = random_generator_negative<T>();
                }
            }
        }
    }
};

/*! \brief  symmetric matrix initialization: */
// for real matrix only
template <typename T>
void hipblas_init_symmetric(vector<T>& A, int N, int lda)
{
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j <= i; ++j)
        {
            A[j + i * lda] = A[i + j * lda] = random_generator<T>();
        }
    }
};

/*! \brief  hermitian matrix initialization: */
// for complex matrix only, the real/imag part would be initialized with the same value
// except the diagonal elment must be real
template <typename T>
void hipblas_init_hermitian(vector<T>& A, int N, int lda)
{
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j <= i; ++j)
        {
            A[j + i * lda] = A[i + j * lda] = random_generator<T>();
            if(i == j)
                A[j + i * lda].y = 0.0;
        }
    }
};

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

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipblas_float_complex -> 'c', hipblas_double_complex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T>
void print_matrix(vector<T> CPU_result, vector<T> GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
        {
            printf("matrix  col %d, row %d, CPU result=%f, GPU result=%f\n",
                   i,
                   j,
                   CPU_result[j + i * lda],
                   GPU_result[j + i * lda]);
        }
}

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

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this hipblas library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
public:
    int M = 128;
    int N = 128;
    int K = 128;

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

    double alpha = 1.0;
    double beta  = 0.0;

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
        M = rhs.M;
        N = rhs.N;
        K = rhs.K;

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

        alpha = rhs.alpha;
        beta  = rhs.beta;

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
};

#endif
