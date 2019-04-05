/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _NORM_H
#define _NORM_H

#include "hipblas.h"

/* =====================================================================
        Norm check: norm(A-B)/norm(A), evaluate relative error
    =================================================================== */

/*!\file
 * \brief compares two results (usually, CPU and GPU results); provides Norm check
 */

/* ========================================Norm Check
 * ==================================================== */

/*! \brief  Template: norm check for general Matrix: float/doubel/complex  */

// see check_norm.cpp for template speciliazation
// use auto as the return type is only allowed in c++14
// convert float/float to double
template <typename T>
double norm_check_general(char norm_type, int M, int N, int lda, T* hCPU, T* hGPU);

/*! \brief  Template: norm check for hermitian/symmetric Matrix: float/double/complex */

template <typename T>
double norm_check_symmetric(char norm_type, char uplo, int N, int lda, T* hCPU, T* hGPU);

#endif
