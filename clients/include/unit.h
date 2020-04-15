/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _UNIT_H
#define _UNIT_H

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

/*! \brief Template: gtest unit compare two matrices float/double/complex */
// Do not put a wrapper over ASSERT_FLOAT_EQ, sincer assert exit the current function NOT the test
// case
// a wrapper will cause the loop keep going
template <typename T>
void unit_check_general(int M, int N, int lda, T* hCPU, T* hGPU);

template <typename T>
void unit_check_general(int M, int N, int batch_count, int lda, int stride_A, T* hCPU, T* hGPU);

template <typename T>
void unit_check_general(int M, int N, int batch_count, int lda, T** hCPU, T** hGPU);

template <typename T>
void unit_check_general(
    int M, int N, int batch_count, int lda, host_vector<T> hCPU[], host_vector<T> hGPU[]);

template <typename T>
void unit_check_error(T error, T tolerance)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(error, tolerance);
#endif
}

template <typename T>
void unit_check_nrm2(T cpu_result, T gpu_result, T tolerance)
{
    T allowable_error = tolerance * std::numeric_limits<T>::epsilon() * cpu_result;
    if(allowable_error == 0)
        allowable_error = tolerance * std::numeric_limits<T>::epsilon();
#ifdef GOOGLE_TEST
    ASSERT_NEAR(cpu_result, gpu_result, allowable_error);
#endif
}

#endif
