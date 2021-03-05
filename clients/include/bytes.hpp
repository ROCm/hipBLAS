/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#ifndef _HIPBLAS_BYTES_H_
#define _HIPBLAS_BYTES_H_

#include "hipblas.h"

/*!\file
 * \brief provides bandwidth measure as byte counts Basic Linear Algebra Subprograms (BLAS) of
 * Level 1, 2, 3. Where possible we are using the values of NOP from the legacy BLAS files
 * [sdcz]blas[23]time.f for byte counts.
 */

/*
 * ===========================================================================
 *    Auxiliary
 * ===========================================================================
 */

/* \brief byte counts of SET/GET_MATRIX/_ASYNC calls done in pairs for timing */
template <typename T>
constexpr double set_get_matrix_gbyte_count(int m, int n)
{
    return (sizeof(T) * m * n * 2.0) / 1e9;
}

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

/* \brief byte counts of ASUM */
template <typename T>
constexpr double asum_gbyte_count(int n)
{
    return (sizeof(T) * n) / 1e9;
}

/* \brief byte counts of AXPY */
template <typename T>
constexpr double axpy_gbyte_count(int n)
{
    return (sizeof(T) * 3.0 * n) / 1e9;
}

/* \brief byte counts of COPY */
template <typename T>
constexpr double copy_gbyte_count(int n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of DOT */
template <typename T>
constexpr double dot_gbyte_count(int n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of NRM2 */
template <typename T>
constexpr double nrm2_gbyte_count(int n)
{
    return (sizeof(T) * n) / 1e9;
}

/* \brief byte counts of SCAL */
template <typename T>
constexpr double scal_gbyte_count(int n)
{
    return (sizeof(T) * 2.0 * n) / 1e9;
}

/* \brief byte counts of SWAP */
template <typename T>
constexpr double swap_gbyte_count(int n)
{
    return (sizeof(T) * 4.0 * n) / 1e9;
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

inline size_t tri_count(int n)
{
    return size_t(n) * (1 + n) / 2;
}

/* \brief byte counts of GEMV */
template <typename T>
constexpr double gemv_gbyte_count(hipblasOperation_t transA, int m, int n)
{
    return (sizeof(T) * (m * n + 2 * (transA == HIPBLAS_OP_N ? n : m))) / 1e9;
}

/* \brief byte counts of GER */
template <typename T>
constexpr double ger_gbyte_count(int m, int n)
{
    return (sizeof(T) * (m * n + m + n)) / 1e9;
}

/* \brief byte counts of HPR */
template <typename T>
constexpr double hpr_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of HPR2 */
template <typename T>
constexpr double hpr2_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + 2.0 * n)) / 1e9;
}

/* \brief byte counts of SYMV */
template <typename T>
constexpr double symv_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of SPMV */
template <typename T>
constexpr double spmv_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of SBMV */
template <typename T>
constexpr double sbmv_gbyte_count(int n, int k)
{
    int k1 = k < n ? k : n - 1;
    return (sizeof(T) * (tri_count(n) - tri_count(n - (k1 + 1)) + n)) / 1e9;
}

/* \brief byte counts of HER */
template <typename T>
constexpr double her_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte counts of HER2 */
template <typename T>
constexpr double her2_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + 2 * n)) / 1e9;
}

/* \brief byte  counts of SYR2 */
template <typename T>
constexpr double syr2_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + 2 * n)) / 1e9;
}

/* \brief byte coutns of TBSV */
template <typename T>
constexpr double tbsv_gbyte_count(int n, int k)
{
    int k1 = k < n ? k : n;
    return (sizeof(T) * (n * k1 - ((k1 * (k1 + 1)) / 2.0) + 2 * n)) / 1e9;
}

/* \brief byte counts of TPSV */
template <typename T>
constexpr double tpsv_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/* \brief byte c ounts or TRSV */
template <typename T>
constexpr double trsv_gbyte_count(int n)
{
    return (sizeof(T) * (tri_count(n) + n)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

/* \brief byte counts of GEMM */
template <typename T>
constexpr double gemm_gbyte_count(int m, int n, int k)
{
    return (sizeof(T) * (m * k + n * k + m * n)) / 1e9;
}

/* \brief byte counts of TRMM */
template <typename T>
constexpr double trmm_gbyte_count(int m, int n, int k)
{
    return (sizeof(T) * (m * n * 2 + k * k / 2)) / 1e9;
}

/* \brief byte counts of TRSM */
template <typename T>
constexpr double trsm_gbyte_count(int m, int n, int k)
{
    return (sizeof(T) * (tri_count(k) + n * m)) / 1e9;
}

/* \brief byte counts of SYRK */
template <typename T>
constexpr double syrk_gbyte_count(int n, int k)
{
    int k1 = k < n ? k : n - 1;
    return (sizeof(T) * (tri_count(n) + n * k)) / 1e9;
}

/* \brief byte counts of HERK */
template <typename T>
constexpr double herk_gbyte_count(int n, int k)
{
    return syrk_gbyte_count<T>(n, k);
}

#endif /* _HIPBLAS_BYTES_H_ */
