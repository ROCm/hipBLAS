/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "unit.h"
#include "hipblas.h"
#include "utility.h"

/* ========================================Gtest Unit Check
 * ==================================================== */

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going

template <>
void unit_check_general(int M, int N, int lda, hipblasHalf* hCPU, hipblasHalf* hGPU)
{
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            float cpu_float = half_to_float(hCPU[i + j * lda]);
            float gpu_float = half_to_float(hGPU[i + j * lda]);
            ASSERT_FLOAT_EQ(cpu_float, gpu_float);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, int lda, float* hCPU, float* hGPU)
{
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_FLOAT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
}

template <>
void unit_check_general(int M, int N, int lda, double* hCPU, double* hGPU)
{
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_DOUBLE_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
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
#pragma unroll
    for(int j = 0; j < N; j++)
    {
#pragma unroll
        for(int i = 0; i < M; i++)
        {
#ifdef GOOGLE_TEST
            ASSERT_EQ(hCPU[i + j * lda], hGPU[i + j * lda]);
#endif
        }
    }
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
