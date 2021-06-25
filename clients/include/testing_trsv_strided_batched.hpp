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

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrsvStridedBatchedFn
        = FORTRAN ? hipblasTrsvStridedBatched<T, true> : hipblasTrsvStridedBatched<T, false>;

    int                M            = argus.M;
    int                incx         = argus.incx;
    int                lda          = argus.lda;
    char               char_uplo    = argus.uplo_option;
    char               char_diag    = argus.diag_option;
    char               char_transA  = argus.transA_option;
    hipblasFillMode_t  uplo         = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA       = char2hipblas_operation(char_transA);
    double             stride_scale = argus.stride_scale;
    int                batch_count  = argus.batch_count;

    int           abs_incx = incx < 0 ? -incx : incx;
    hipblasStride strideA  = lda * M * stride_scale;
    hipblasStride stridex  = abs_incx * M * stride_scale;
    size_t        size_A   = size_t(strideA) * batch_count;
    size_t        size_x   = size_t(stridex) * batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || lda < M || !incx || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count || !M || !lda)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);

    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    double             gpu_time_used, hipblas_error, cumulative_hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda, strideA, batch_count);
    hipblas_init<T>(hx, 1, M, abs_incx, stridex, batch_count);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAb  = hA.data() + b * strideA;
        T* AATb = AAT.data() + b * strideA;
        T* hbb  = hb.data() + b * stridex;
        //  calculate AAT = hA * hA ^ T
        cblas_gemm<T>(
            HIPBLAS_OP_N, HIPBLAS_OP_T, M, M, M, (T)1.0, hAb, lda, hAb, lda, (T)0.0, AATb, lda);

        //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
        for(int i = 0; i < M; i++)
        {
            T t = 0.0;
            for(int j = 0; j < M; j++)
            {
                hAb[i + j * lda] = AATb[i + j * lda];
                t += abs(AATb[i + j * lda]);
            }
            hAb[i + i * lda] = t;
        }
        //  calculate Cholesky factorization of SPD matrix hA
        cblas_potrf<T>(char_uplo, M, hAb, lda);

        //  make hA unit diagonal if diag == rocblas_diagonal_unit
        if(char_diag == 'U' || char_diag == 'u')
        {
            if('L' == char_uplo || 'l' == char_uplo)
                for(int i = 0; i < M; i++)
                {
                    T diag = hAb[i + i * lda];
                    for(int j = 0; j <= i; j++)
                        hAb[i + j * lda] = hAb[i + j * lda] / diag;
                }
            else
                for(int j = 0; j < M; j++)
                {
                    T diag = hAb[j + j * lda];
                    for(int i = 0; i <= j; i++)
                        hAb[i + j * lda] = hA[b + j * lda] / diag;
                }
        }

        // Calculate hb = hA*hx;
        cblas_trmv<T>(uplo, transA, diag, M, hAb, lda, hbb, incx);
    }

    hx_or_b_1 = hb;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrsvStridedBatchedFn(
            handle, uplo, transA, diag, M, dA, lda, strideA, dx_or_b, incx, stridex, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(
                M, abs_incx, hx.data() + b * stridex, hx_or_b_1.data() + b * stridex));
            if(argus.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * M;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrsvStridedBatchedFn(handle,
                                                            uplo,
                                                            transA,
                                                            diag,
                                                            M,
                                                            dA,
                                                            lda,
                                                            strideA,
                                                            dx_or_b,
                                                            incx,
                                                            stridex,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_lda,
                      e_incx,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trsv_gflop_count<T>(M),
                         trsv_gbyte_count<T>(M),
                         cumulative_hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
