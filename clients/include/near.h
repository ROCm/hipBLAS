/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _NEAR_H
#define _NEAR_H

#include "hipblas.h"
#include "hipblas_vector.hpp"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

/* =====================================================================

    Google Unit check: ASSERT_EQ( elementof(A), elementof(B))

   =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Google Unit check.
 */

/* ========================================Gtest Unit Check
 * ==================================================== */

// sqrt(0.5) factor for complex cutoff calculations
constexpr double sqrthalf = 0.7071067811865475244;

/*! \brief Template: gtest near compare two matrices float/double/complex */
template <typename T>
void near_check_general(int M, int N, int lda, T* hCPU, T* hGPU, double abs_error);

template <typename T>
void near_check_general(
    int M, int N, int lda, host_vector<T> hCPU, host_vector<T> hGPU, double abs_error);

template <typename T>
void near_check_general(
    int M, int N, int batch_count, int lda, int stride_A, T* hCPU, T* hGPU, double abs_error);

template <typename T>
void near_check_general(
    int M, int N, int batch_count, int lda, T** hCPU, T** hGPU, double abs_error);

template <typename T>
void near_check_general(int            M,
                        int            N,
                        int            batch_count,
                        int            lda,
                        host_vector<T> hCPU[],
                        host_vector<T> hGPU[],
                        double         abs_error);

#endif
