/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "unit.h"
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
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)
#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)
#else
#define UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, UNIT_ASSERT_EQ)      \
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
                        UNIT_ASSERT_EQ(hCPU[i + j * lda + k * strideA],              \
                                       hGPU[i + j * lda + k * strideA]);             \
                    }                                                                \
    } while(0)
#define UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, UNIT_ASSERT_EQ)            \
    do                                                                              \
    {                                                                               \
        for(size_t k = 0; k < batch_count; k++)                                     \
            for(size_t j = 0; j < N; j++)                                           \
                for(size_t i = 0; i < M; i++)                                       \
                    if(hipblas_isnan(hCPU[k][i + j * lda]))                         \
                    {                                                               \
                        ASSERT_TRUE(hipblas_isnan(hGPU[k][i + j * lda]));           \
                    }                                                               \
                    else                                                            \
                    {                                                               \
                        UNIT_ASSERT_EQ(hCPU[k][i + j * lda], hGPU[k][i + j * lda]); \
                    }                                                               \
    } while(0)
#endif

#define ASSERT_HALF_EQ(a, b) ASSERT_FLOAT_EQ(half_to_float(a), half_to_float(b))
#define ASSERT_BFLOAT16_EQ(a, b) ASSERT_FLOAT_EQ(bfloat16_to_float(a), bfloat16_to_float(b))

#define ASSERT_FLOAT_COMPLEX_EQ(a, b) \
    do                                \
    {                                 \
        ASSERT_FLOAT_EQ(a.x, b.x);    \
        ASSERT_FLOAT_EQ(a.y, b.y);    \
    } while(0)

#define ASSERT_DOUBLE_COMPLEX_EQ(a, b) \
    do                                 \
    {                                  \
        ASSERT_DOUBLE_EQ(a.x, b.x);    \
        ASSERT_DOUBLE_EQ(a.y, b.y);    \
    } while(0)

template <>
void unit_check_general(int M, int N, int lda, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int M, int N, int lda, hipblasBfloat16* hCPU, hipblasBfloat16* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(int M, int N, int lda, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int M, int N, int lda, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int M, int N, int lda, hipComplex* hCPU, hipComplex* hGPU)
{
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_FLOAT_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, int lda, hipDoubleComplex* hCPU, hipDoubleComplex* hGPU)
{
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].x, hGPU[i + j * lda].x);
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda].y, hGPU[i + j * lda].y);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, int lda, int* hCPU, int* hGPU)
{
    UNIT_CHECK(M, N, 1, lda, 0, hCPU, hGPU, ASSERT_EQ);
}

// batched checks
template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, hipblasHalf** hCPU, hipblasHalf** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, hipblasBfloat16** hCPU, hipblasBfloat16** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(int M, int N, int batch_count, int lda, float** hCPU, float** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(int M, int N, int batch_count, int lda, double** hCPU, double** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int M, int N, int batch_count, int lda, int** hCPU, int** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, hipComplex** hCPU, hipComplex** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, hipDoubleComplex** hCPU, hipDoubleComplex** hGPU)
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

// batched checks for host_vector[]s
template <>
void unit_check_general(int                      M,
                        int                      N,
                        int                      batch_count,
                        int                      lda,
                        host_vector<hipblasHalf> hCPU[],
                        host_vector<hipblasHalf> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int                          M,
                        int                          N,
                        int                          batch_count,
                        int                          lda,
                        host_vector<hipblasBfloat16> hCPU[],
                        host_vector<hipblasBfloat16> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, host_vector<int> hCPU[], host_vector<int> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, host_vector<float> hCPU[], host_vector<float> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, host_vector<double> hCPU[], host_vector<double> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(int                     M,
                        int                     N,
                        int                     batch_count,
                        int                     lda,
                        host_vector<hipComplex> hCPU[],
                        host_vector<hipComplex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int                           M,
                        int                           N,
                        int                           batch_count,
                        int                           lda,
                        host_vector<hipDoubleComplex> hCPU[],
                        host_vector<hipDoubleComplex> hGPU[])
{
    UNIT_CHECK_B(M, N, batch_count, lda, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

// strided_batched checks
template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, int strideA, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_HALF_EQ);
}

template <>
void unit_check_general(int              M,
                        int              N,
                        int              batch_count,
                        int              lda,
                        int              strideA,
                        hipblasBfloat16* hCPU,
                        hipblasBfloat16* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_BFLOAT16_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, int strideA, float* hCPU, float* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, int strideA, double* hCPU, double* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_EQ);
}

template <>
void unit_check_general(
    int M, int N, int batch_count, int lda, int strideA, hipComplex* hCPU, hipComplex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_FLOAT_COMPLEX_EQ);
}

template <>
void unit_check_general(int               M,
                        int               N,
                        int               batch_count,
                        int               lda,
                        int               strideA,
                        hipDoubleComplex* hCPU,
                        hipDoubleComplex* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_DOUBLE_COMPLEX_EQ);
}

template <>
void unit_check_general(int M, int N, int batch_count, int lda, int strideA, int* hCPU, int* hGPU)
{
    UNIT_CHECK(M, N, batch_count, lda, strideA, hCPU, hGPU, ASSERT_EQ);
}

/* ========================================Gtest Unit Check TRSM
 * ==================================================== */

/*! \brief Template: determine trsm error tolerance: 1e-5 and 1e-12 respectively for float/double
 * precision */

template <>
float get_trsm_tolerance()
{
    return 5 * 1e-5;
}

template <>
double get_trsm_tolerance()
{
    return 1e-12;
}

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

// trsm has division, must use near to suppress the false failure
template <>
void unit_check_trsm(int M, int N, int lda, double hGPU, float tolerance)
{

#ifdef GOOGLE_TEST
    ASSERT_LE(hGPU, tolerance);
#endif
}

template <>
void unit_check_trsm(int M, int N, int lda, double hGPU, double tolerance)
{

#ifdef GOOGLE_TEST
    ASSERT_LE(hGPU, tolerance);
#endif
}
