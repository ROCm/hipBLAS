/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

// reference implementation of syrkx. This function is not part of legacy BLAS.
template <typename T>
void syrkx_reference(hipblasFillMode_t  uplo,
                     hipblasOperation_t trans,
                     int                n,
                     int                k,
                     T                  alpha,
                     T*                 a,
                     int                lda,
                     T*                 b,
                     int                ldb,
                     T                  beta,
                     T*                 c,
                     int                ldc)
{
    int a_s1 = HIPBLAS_OP_N == trans ? 1 : lda;
    int a_s2 = HIPBLAS_OP_N == trans ? lda : 1;
    int b_s1 = HIPBLAS_OP_N == trans ? 1 : ldb;
    int b_s2 = HIPBLAS_OP_N == trans ? ldb : 1;
    int c_s1 = 1;
    int c_s2 = ldc;

    // argument error
    int nrow = trans == HIPBLAS_OP_N ? n : k;
    if(n < 0)
    {
        std::cout << "ERROR: syrkx_reference n < 0" << std::endl;
        return;
    }
    if(k < 0)
    {
        std::cout << "ERROR: syrk_reference k < 0" << std::endl;
        return;
    }
    if(n > ldc)
    {
        std::cout << "ERROR: syrk_reference n > ldc" << std::endl;
        return;
    }
    if(nrow > lda)
    {
        std::cout << "ERROR: syrk_reference nrow > lda" << std::endl;
        return;
    }
    if(nrow > ldb)
    {
        std::cout << "ERROR: syrk_reference nrow > ldb" << std::endl;
        return;
    }

    // quick return
    if((n == 0) || (((alpha == 0) || (k == 0)) && (beta == 1)))
        return;

    // rank kx update with special cases for alpha == 0, beta == 0
    for(int i1 = 0; i1 < n; i1++)
    {
        int i2_start = HIPBLAS_FILL_MODE_LOWER == uplo ? 0 : i1;
        int i2_end   = HIPBLAS_FILL_MODE_LOWER == uplo ? i1 + 1 : n;
        for(int i2 = i2_start; i2 < i2_end; i2++)
        {
            if(alpha == 0 && beta == 0)
            {
                c[i1 * c_s1 + i2 * c_s2] = 0.0;
            }
            else if(alpha == 0)
            {
                c[i1 * c_s1 + i2 * c_s2] *= beta;
            }
            else
            {
                T t = 0;
                for(int i3 = 0; i3 < k; i3++)
                {
                    t += a[i1 * a_s1 + i3 * a_s2] * b[i2 * b_s1 + i3 * b_s2];
                }
                c[i1 * c_s1 + i2 * c_s2] = beta * c[i1 * c_s1 + i2 * c_s2] + alpha * t;
            }
        }
    }
    return;
}
