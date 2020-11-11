/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _NORM_H
#define _NORM_H

#include "hipblas.h"
#include "hipblas_vector.hpp"

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

template <typename T>
double norm_check_general(char           norm_type,
                          int            M,
                          int            N,
                          int            lda,
                          host_vector<T> hCPU[],
                          host_vector<T> hGPU[],
                          int            batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(int i = 0; i < batch_count; i++)
    {
        auto index = i;

        auto error = norm_check_general<T>(norm_type, M, N, lda, hCPU[index], hGPU[index]);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

/* ============== Norm Check for strided_batched case ============= */
template <typename T>
double norm_check_general(
    char norm_type, int M, int N, int lda, ptrdiff_t stride_a, T* hCPU, T* hGPU, int batch_count)
{
    // norm type can be O', 'I', 'F', 'o', 'i', 'f' for one, infinity or Frobenius norm
    // one norm is max column sum
    // infinity norm is max row sum
    // Frobenius is l2 norm of matrix entries
    //
    // use triangle inequality ||a+b|| <= ||a|| + ||b|| to calculate upper limit for Frobenius norm
    // of strided batched matrix

    double cumulative_error = 0.0;

    for(size_t i = 0; i < batch_count; i++)
    {
        auto index = i * stride_a;

        auto error = norm_check_general(norm_type, M, N, lda, hCPU + index, hGPU + index);

        if(norm_type == 'F' || norm_type == 'f')
        {
            cumulative_error += error;
        }
        else if(norm_type == 'O' || norm_type == 'o' || norm_type == 'I' || norm_type == 'i')
        {
            cumulative_error = cumulative_error > error ? cumulative_error : error;
        }
    }

    return cumulative_error;
}

#endif
