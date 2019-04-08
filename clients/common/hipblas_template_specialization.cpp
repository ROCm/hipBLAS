/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************/

#include "hipblas.h"
#include "hipblas.hpp"
#include <typeinfo>

/*!\file
 * \brief provide template functions interfaces to ROCBLAS C89 interfaces
*/

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */
// scal
template <>
hipblasStatus_t
    hipblasScal<float>(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx)
{

    return hipblasSscal(handle, n, alpha, x, incx);
}

template <>
hipblasStatus_t
    hipblasScal<double>(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx)
{

    return hipblasDscal(handle, n, alpha, x, incx);
}
/*
    template<>
    hipblasStatus_t
    hipblasScal<hipComplex>(hipblasHandle_t handle,
        int n,
        const hipComplex *alpha,
        hipComplex *x, int incx){

        return hipblasCscal(handle, n, alpha, x, incx);
    }

    template<>
    hipblasStatus_t
    hipblasScal<hipDoubleComplex>(hipblasHandle_t handle,
        int n,
        const hipDoubleComplex *alpha,
        hipDoubleComplex *x, int incx){

        return hipblasZscal(handle, n, alpha, x, incx);
    }
*/
/*
    //swap
    template<>
    hipblasStatus_t
    hipblasSwap<float>(    hipblasHandle_t handle, int n,
                            float *x, int incx,
                            float *y, int incy)
    {
        return hipblasSwap(handle, n, x, incx, y, incy);
    }

    template<>
    hipblasStatus_t
    hipblasSwap<double>(   hipblasHandle_t handle, int n,
                            double *x, int incx,
                            double *y, int incy)
    {
        return hipblasDswap(handle, n, x, incx, y, incy);
    }

    template<>
    hipblasStatus_t
    hipblasSwap<hipComplex>(    hipblasHandle_t handle, int n,
                            hipComplex *x, int incx,
                            hipComplex *y, int incy)
    {
        return hipblasCswap(handle, n, x, incx, y, incy);
    }

    template<>
    hipblasStatus_t
    hipblasSwap<hipDoubleComplex>(    hipblasHandle_t handle, int n,
                            hipDoubleComplex *x, int incx,
                            hipDoubleComplex *y, int incy)
    {
        return hipblasZswap(handle, n, x, incx, y, incy);
    }
*/
// copy
template <>
hipblasStatus_t
    hipblasCopy<float>(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    return hipblasScopy(handle, n, x, incx, y, incy);
}

template <>
hipblasStatus_t hipblasCopy<double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy)
{
    return hipblasDcopy(handle, n, x, incx, y, incy);
}
/*
    template<>
    hipblasStatus_t
    hipblasCopy<hipComplex>(    hipblasHandle_t handle, int n,
                            const hipComplex *x, int incx,
                            hipComplex *y, int incy)
    {
        return hipblasCcopy(handle, n, x, incx, y, incy);
    }

    template<>
    hipblasStatus_t
    hipblasCopy<hipDoubleComplex>(    hipblasHandle_t handle, int n,
                            const hipDoubleComplex *x, int incx,
                            hipDoubleComplex *y, int incy)
    {
        return hipblasZcopy(handle, n, x, incx, y, incy);
    }
*/
// dot
template <>
hipblasStatus_t hipblasDot<float>(hipblasHandle_t handle,
                                  int             n,
                                  const float*    x,
                                  int             incx,
                                  const float*    y,
                                  int             incy,
                                  float*          result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}

template <>
hipblasStatus_t hipblasDot<double>(hipblasHandle_t handle,
                                   int             n,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}
/*
    template<>
    hipblasStatus_t
    hipblasDot<hipComplex>(    hipblasHandle_t handle, int n,
                            const hipComplex *x, int incx,
                            const hipComplex *y, int incy,
                            hipComplex *result)
    {
        return hipblasCdotu(handle, n, x, incx, y, incy, result);
    }

    template<>
    hipblasStatus_t
    hipblasDot<hipDoubleComplex>(    hipblasHandle_t handle, int n,
                            const hipDoubleComplex *x, int incx,
                            const hipDoubleComplex *y, int incy,
                            hipDoubleComplex *result)
    {
        return hipblasZdotu(handle, n, x, incx, y, incy, result);
    }
*/

// asum
template <>
hipblasStatus_t hipblasAsum<float, float>(
    hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{

    return hipblasSasum(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasAsum<double, double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{

    return hipblasDasum(handle, n, x, incx, result);
}
/*
    template<>
    hipblasStatus_t
    hipblasAsum<hipComplex, float>(hipblasHandle_t handle,
        int n,
        const hipComplex *x, int incx,
        float *result){

        return hipblasScasum(handle, n, x, incx, result);
    }
*/

// nrm2
template <>
hipblasStatus_t hipblasNrm2<float, float>(
    hipblasHandle_t handle, int n, const float* x, int incx, float* result)
{

    return hipblasSnrm2(handle, n, x, incx, result);
}

template <>
hipblasStatus_t hipblasNrm2<double, double>(
    hipblasHandle_t handle, int n, const double* x, int incx, double* result)
{

    return hipblasDnrm2(handle, n, x, incx, result);
}
/*
    template<>
    hipblasStatus_t
    hipblasNrm2<hipComplex, float>(hipblasHandle_t handle,
        int n,
        const hipComplex *x, int incx,
        float *result){

        return hipblasScnrm2(handle, n, x, incx, result);
    }

    template<>
    hipblasStatus_t
    hipblasNrm2<hipDoubleComplex, double>(hipblasHandle_t handle,
        int n,
        const hipDoubleComplex *x, int incx,
        double *result){

        return hipblasDznrm2(handle, n, x, incx, result);
    }
*/

/*
    //amin
    template<>
    hipblasStatus_t
    hipblasAmin<float>(hipblasHandle_t handle,
        int n,
        const float *x, int incx,
        int *result){

        return hipblasSamin(handle, n, x, incx, result);
    }

    template<>
    hipblasStatus_t
    hipblasAmin<double>(hipblasHandle_t handle,
        int n,
        const double *x, int incx,
        int *result){

        return hipblasDamin(handle, n, x, incx, result);
    }

    template<>
    hipblasStatus_t
    hipblasAmin<hipComplex>(hipblasHandle_t handle,
        int n,
        const hipComplex *x, int incx,
        int *result){

        return hipblasScamin(handle, n, x, incx, result);
    }

    template<>
    hipblasStatus_t
    hipblasAmin<hipDoubleComplex>(hipblasHandle_t handle,
        int n,
        const hipDoubleComplex *x, int incx,
        int *result){

        return hipblasDzamin(handle, n, x, incx, result);
    }
*/
// amax
template <>
hipblasStatus_t
    hipblasIamax<float>(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return hipblasIsamax(handle, n, x, incx, result);
}

template <>
hipblasStatus_t
    hipblasIamax<double>(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
{
    return hipblasIdamax(handle, n, x, incx, result);
}

/*
    template<>
    hipblasStatus_t
    hipblasAmax<hipComplex>(hipblasHandle_t handle,
        int n,
        const hipComplex *x, int incx,
        int *result){

        return hipblasScamax(handle, n, x, incx, result);
    }

    template<>
    hipblasStatus_t
    hipblasAmax<hipDoubleComplex>(hipblasHandle_t handle,
        int n,
        const hipDoubleComplex *x, int incx,
        int *result){

        return hipblasDzamax(handle, n, x, incx, result);
    }
*/
/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

template <>
hipblasStatus_t hipblasGemv<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float*       A,
                                   int                lda,
                                   const float*       x,
                                   int                incx,
                                   const float*       beta,
                                   float*             y,
                                   int                incy)
{
    return hipblasSgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGemv<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      x,
                                    int                incx,
                                    const double*      beta,
                                    double*            y,
                                    int                incy)
{
    return hipblasDgemv(handle, transA, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
hipblasStatus_t hipblasGer<float>(hipblasHandle_t handle,
                                  int             m,
                                  int             n,
                                  const float*    alpha,
                                  const float*    x,
                                  int             incx,
                                  const float*    y,
                                  int             incy,
                                  float*          A,
                                  int             lda)
{

    return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
hipblasStatus_t hipblasGer<double>(hipblasHandle_t handle,
                                   int             m,
                                   int             n,
                                   const double*   alpha,
                                   const double*   x,
                                   int             incx,
                                   const double*   y,
                                   int             incy,
                                   double*         A,
                                   int             lda)
{

    return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

//

/*
    template<>
    hipblasStatus_t
    hipblasTrtri<float>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        float *A, int lda,
        float *invA, int ldinvA){
        return hipblasStrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri<double>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        double *A, int lda,
        double *invA, int ldinvA){
        return hipblasDtrtri(handle, uplo, diag, n, A, lda, invA, ldinvA);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri_batched<float>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        float *A, int lda, int bsa,
        float *invA, int ldinvA, int bsinvA,
        int batch_count){
        return hipblasStrtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA,
   batch_count);
    }

    template<>
    hipblasStatus_t
    hipblasTrtri_batched<double>(hipblasHandle_t handle,
        hipblasFillMode_t uplo,
        hipblasDiagType_t diag,
        int n,
        double *A, int lda, int bsa,
        double *invA, int ldinvA, int bsinvA,
        int batch_count){
        return hipblasDtrtri_batched(handle, uplo, diag, n, A, lda, bsa, invA, ldinvA, bsinvA,
   batch_count);
    }
*/

template <>
hipblasStatus_t hipblasGemm<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   int                k,
                                   const float*       alpha,
                                   const float*       A,
                                   int                lda,
                                   const float*       B,
                                   int                ldb,
                                   const float*       beta,
                                   float*             C,
                                   int                ldc)
{
    return hipblasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    hipblasOperation_t transB,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      B,
                                    int                ldb,
                                    const double*      beta,
                                    double*            C,
                                    int                ldc)
{
    return hipblasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 int                k,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                bsa,
                                                 const float*       B,
                                                 int                ldb,
                                                 int                bsb,
                                                 const float*       beta,
                                                 float*             C,
                                                 int                ldc,
                                                 int                bsc,
                                                 int                batch_count)
{

    return hipblasSgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  hipblasOperation_t transB,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                bsa,
                                                  const double*      B,
                                                  int                ldb,
                                                  int                bsb,
                                                  const double*      beta,
                                                  double*            C,
                                                  int                ldc,
                                                  int                bsc,
                                                  int                batch_count)
{

    return hipblasDgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<float>(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          int                k,
                                          const float*       alpha,
                                          const float*       A[],
                                          int                lda,
                                          const float*       B[],
                                          int                ldb,
                                          const float*       beta,
                                          float*             C[],
                                          int                ldc,
                                          int                batch_count)
{
    return hipblasSgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasGemmBatched<double>(hipblasHandle_t    handle,
                                           hipblasOperation_t transA,
                                           hipblasOperation_t transB,
                                           int                m,
                                           int                n,
                                           int                k,
                                           const double*      alpha,
                                           const double*      A[],
                                           int                lda,
                                           const double*      B[],
                                           int                ldb,
                                           const double*      beta,
                                           double*            C[],
                                           int                ldc,
                                           int                batch_count)
{
    return hipblasDgemmBatched(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch_count);
}

template <>
hipblasStatus_t hipblasTrsm<float>(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   float*             A,
                                   int                lda,
                                   float*             B,
                                   int                ldb)
{
    return hipblasStrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasTrsm<double>(hipblasHandle_t    handle,
                                    hipblasSideMode_t  side,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    hipblasDiagType_t  diag,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    double*            A,
                                    int                lda,
                                    double*            B,
                                    int                ldb)
{
    return hipblasDtrsm(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
hipblasStatus_t hipblasGeam<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   const float*       alpha,
                                   const float*       A,
                                   int                lda,
                                   const float*       beta,
                                   const float*       B,
                                   int                ldb,
                                   float*             C,
                                   int                ldc)
{
    return hipblasSgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

template <>
hipblasStatus_t hipblasGeam<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    hipblasOperation_t transB,
                                    int                m,
                                    int                n,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      beta,
                                    const double*      B,
                                    int                ldb,
                                    double*            C,
                                    int                ldc)
{
    return hipblasDgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
