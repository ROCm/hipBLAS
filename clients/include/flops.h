/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#pragma once
#ifndef _ROCBLAS_FLOPS_H_
#define _ROCBLAS_FLOPS_H_

#include "hipblas.h"
#include <typeinfo>

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 2 and
 * 3. No flops counts for Level 1 BLAS
*/

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMV */
template <typename T>
double gemv_gflop_count(int m, int n)
{
    return (2.0 * m * n) / 1e9;
}

/* \brief floating point counts of SY(HE)MV */
template <typename T>
double symv_gflop_count(int n)
{
    return (2.0 * n * n) / 1e9;
}

/* \brief floating point counts of GER */
template <typename T>
double ger_gflop_count(int m, int n)
{
    return (2.0 * m * n) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief floating point counts of GEMM */
template <typename T>
double gemm_gflop_count(int m, int n, int k)
{
    return (2.0 * m * n * k) / 1e9;
}

/* \brief floating point counts of TRSM */
template <typename T>
double trsm_gflop_count(int m, int n, int k)
{
    return (1.0 * m * n * (k + 1)) / 1e9;
}

/* \brief floating point counts of TRTRI */
template <typename T>
double trtri_gflop_count(int n)
{
    return (1.0 * n * n * n) / 3.0 / 1e9;
}

#endif /* _ROCBLAS_FLOPS_H_ */
