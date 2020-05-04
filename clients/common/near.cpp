/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "near.h"
#include "hipblas.h"
#include "hipblas_vector.hpp"
#include "utility.h"

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

#ifndef GOOGLE_TEST
#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)
#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)
#else

#define NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, err, NEAR_ASSERT)    \
    do                                                                               \
    {                                                                                \
        for(size_t k = 0; k < batch_count; k++)                                      \
            for(size_t j = 0; j < N; j++)                                            \
                for(size_t i = 0; i < M; i++)                                        \
                    if(hipblas_isnan(hCPU[i + j * lda + k * strideA]))               \
                    {                                                                \
                        ASSERT_TRUE(hipblas_isnan(hGPU[i + j * lda + k * strideA])); \
                    }                                                                \
                    else                                                             \
                    {                                                                \
                        NEAR_ASSERT(hCPU[i + j * lda + k * strideA],                 \
                                    hGPU[i + j * lda + k * strideA],                 \
                                    err);                                            \
                    }                                                                \
    } while(0)

#define NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, err, NEAR_ASSERT)            \
    do                                                                                \
    {                                                                                 \
        for(size_t k = 0; k < batch_count; k++)                                       \
            for(size_t j = 0; j < N; j++)                                             \
                for(size_t i = 0; i < M; i++)                                         \
                    if(hipblas_isnan(hCPU[k][i + j * lda]))                           \
                    {                                                                 \
                        ASSERT_TRUE(hipblas_isnan(hGPU[k][i + j * lda]));             \
                    }                                                                 \
                    else                                                              \
                    {                                                                 \
                        NEAR_ASSERT(hCPU[k][i + j * lda], hGPU[k][i + j * lda], err); \
                    }                                                                 \
    } while(0)

#endif

#define NEAR_ASSERT_HALF(a, b, err) ASSERT_NEAR(float(a), float(b), err)

#define NEAR_ASSERT_COMPLEX(a, b, err)          \
    do                                          \
    {                                           \
        auto ta = (a), tb = (b);                \
        ASSERT_NEAR(ta.real(), tb.real(), err); \
        ASSERT_NEAR(ta.imag(), tb.imag(), err); \
    } while(0)

template <>
void near_check_general(int M, int N, int lda, float* hCPU, float* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int M, int N, int lda, double* hCPU, double* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(
    int M, int N, int lda, hipblasHalf* hCPU, hipblasHalf* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(
    int M, int N, int lda, hipblasComplex* hCPU, hipblasComplex* hGPU, double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(
    int M, int N, int lda, hipblasDoubleComplex* hCPU, hipblasDoubleComplex* hGPU, double abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, 1, lda, 0, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(
    int M, int N, int batch_count, int lda, int strideA, float* hCPU, float* hGPU, double abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int     M,
                        int     N,
                        int     batch_count,
                        int     lda,
                        int     strideA,
                        double* hCPU,
                        double* hGPU,
                        double  abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int          M,
                        int          N,
                        int          batch_count,
                        int          lda,
                        int          strideA,
                        hipblasHalf* hCPU,
                        hipblasHalf* hGPU,
                        double       abs_error)
{
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int             M,
                        int             N,
                        int             batch_count,
                        int             lda,
                        int             strideA,
                        hipblasComplex* hCPU,
                        hipblasComplex* hGPU,
                        double          abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int                   M,
                        int                   N,
                        int                   batch_count,
                        int                   lda,
                        int                   strideA,
                        hipblasDoubleComplex* hCPU,
                        hipblasDoubleComplex* hGPU,
                        double                abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int                      M,
                        int                      N,
                        int                      batch_count,
                        int                      lda,
                        host_vector<hipblasHalf> hCPU[],
                        host_vector<hipblasHalf> hGPU[],
                        double                   abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_HALF);
}

template <>
void near_check_general(int                M,
                        int                N,
                        int                batch_count,
                        int                lda,
                        host_vector<float> hCPU[],
                        host_vector<float> hGPU[],
                        double             abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int                 M,
                        int                 N,
                        int                 batch_count,
                        int                 lda,
                        host_vector<double> hCPU[],
                        host_vector<double> hGPU[],
                        double              abs_error)
{
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, ASSERT_NEAR);
}

template <>
void near_check_general(int                         M,
                        int                         N,
                        int                         batch_count,
                        int                         lda,
                        host_vector<hipblasComplex> hCPU[],
                        host_vector<hipblasComplex> hGPU[],
                        double                      abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}

template <>
void near_check_general(int                               M,
                        int                               N,
                        int                               batch_count,
                        int                               lda,
                        host_vector<hipblasDoubleComplex> hCPU[],
                        host_vector<hipblasDoubleComplex> hGPU[],
                        double                            abs_error)
{
    abs_error *= sqrthalf;
    NEAR_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, abs_error, NEAR_ASSERT_COMPLEX);
}
