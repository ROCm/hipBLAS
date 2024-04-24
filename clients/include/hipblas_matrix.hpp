/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "hipblas_init.hpp"

//Temporary could be deleted
/*template <typename T>
void hipblas_init_matrix(const char      uplo,
                         T               rand_gen(),
                         host_vector<T>& A,
                         int64_t          M,
                         int64_t          N,
                         int64_t          lda,
                         hipblasStride   stride      = 0,
                         int64_t         batch_count = 1)
{
    hipblas_fill_matrix_type(
        hipblas_general_matrix, uplo, rand_gen(), A, M, N, lda, stride, batch_count);
}

template <typename T>
void hipblas_init_matrix(
    const char uplo, T rand_gen(), host_batch_vector<T>& A, size_t M, size_t N, size_t lda)
{
    for(int64_t b = 0; b < A.batch_count(); b++)
        hipblas_fill_matrix_type(
            hipblas_general_matrix, uplo, rand_gen(), (T*)A[b], M, N, lda, 0, 1);
}*/

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_matrix(host_matrix<T>&         hA,
                                const Arguments&        arg,
                                hipblas_client_nan_init nan_init,
                                hipblas_matrix_type     matrix_type,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}

//!
//! @brief Initialize a host batch matrix.
//! @param hA The host batch matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_matrix(host_batch_matrix<T>&   hA,
                                const Arguments&        arg,
                                hipblas_client_nan_init nan_init,
                                hipblas_matrix_type     matrix_type,
                                bool                    seedReset        = false,
                                bool                    alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}

//!
//! @brief Initialize a host strided batch matrix.
//! @param hA The host strided batch matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the hipblas_client_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T>
inline void hipblas_init_matrix(host_strided_batch_matrix<T>& hA,
                                const Arguments&              arg,
                                hipblas_client_nan_init       nan_init,
                                hipblas_matrix_type           matrix_type,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    if(seedReset)
        hipblas_seedrand();

    if(nan_init == hipblas_client_alpha_sets_nan && hipblas_isnan(arg.alpha))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == hipblas_client_beta_sets_nan && hipblas_isnan(arg.beta))
    {
        hipblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::hpl)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::rand_int)
    {
        if(alternating_sign)
            hipblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            hipblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == hipblas_initialization::trig_float)
    {
        hipblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
}
