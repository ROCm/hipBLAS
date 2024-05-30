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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

#define TRSM_BLOCK 128

/* ============================================================================================ */

using hipblasTrsmExModel
    = ArgumentModel<e_a_type, e_side, e_uplo, e_transA, e_diag, e_M, e_N, e_alpha, e_lda, e_ldb>;

inline void testname_trsm_ex(const Arguments& arg, std::string& name)
{
    hipblasTrsmExModel{}.test_name(arg, name);
}

template <typename T>
void testing_trsm_ex_bad_arg(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsmExFn = FORTRAN ? hipblasTrsmExFortran : hipblasTrsmEx;

    hipblasLocalHandle handle(arg);

    int64_t            M           = 101;
    int64_t            N           = 100;
    int64_t            lda         = 102;
    int64_t            ldb         = 103;
    hipblasSideMode_t  side        = HIPBLAS_SIDE_LEFT;
    hipblasFillMode_t  uplo        = HIPBLAS_FILL_MODE_LOWER;
    hipblasOperation_t transA      = HIPBLAS_OP_N;
    hipblasDiagType_t  diag        = HIPBLAS_DIAG_NON_UNIT;
    hipblasDatatype_t  computeType = arg.compute_type;

    int64_t K        = side == HIPBLAS_SIDE_LEFT ? M : N;
    int64_t invAsize = TRSM_BLOCK * K;

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);
    device_vector<T> dinvA(TRSM_BLOCK, TRSM_BLOCK, K);

    device_vector<T> d_alpha(1), d_zero(1);
    const T          h_alpha(1), h_zero(0);

    const T* alpha = &h_alpha;
    const T* zero  = &h_zero;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(nullptr,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              HIPBLAS_SIDE_BOTH,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              side,
                                              HIPBLAS_FILL_MODE_FULL,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              side,
                                              uplo,
                                              (hipblasOperation_t)HIPBLAS_FILL_MODE_FULL,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_INVALID_ENUM);
        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              (hipblasDiagType_t)HIPBLAS_FILL_MODE_FULL,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_INVALID_ENUM);

        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              alpha,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              HIPBLAS_R_16F),
                              HIPBLAS_STATUS_NOT_SUPPORTED);

        EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                              side,
                                              uplo,
                                              transA,
                                              diag,
                                              M,
                                              N,
                                              nullptr,
                                              dA,
                                              lda,
                                              dB,
                                              ldb,
                                              dinvA,
                                              invAsize,
                                              computeType),
                              HIPBLAS_STATUS_INVALID_VALUE);

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                                  side,
                                                  uplo,
                                                  transA,
                                                  diag,
                                                  M,
                                                  N,
                                                  alpha,
                                                  nullptr,
                                                  lda,
                                                  dB,
                                                  ldb,
                                                  dinvA,
                                                  invAsize,
                                                  computeType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_HIPBLAS_STATUS(hipblasTrsmExFn(handle,
                                                  side,
                                                  uplo,
                                                  transA,
                                                  diag,
                                                  M,
                                                  N,
                                                  alpha,
                                                  dA,
                                                  lda,
                                                  nullptr,
                                                  ldb,
                                                  dinvA,
                                                  invAsize,
                                                  computeType),
                                  HIPBLAS_STATUS_INVALID_VALUE);
        }

        // If alpha == 0, then A can be nullptr
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            N,
                                            zero,
                                            nullptr,
                                            lda,
                                            dB,
                                            ldb,
                                            dinvA,
                                            invAsize,
                                            computeType));

        // If M == 0 || N == 0, can have nullptrs
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            0,
                                            N,
                                            nullptr,
                                            nullptr,
                                            lda,
                                            nullptr,
                                            ldb,
                                            nullptr,
                                            invAsize,
                                            computeType));
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            0,
                                            nullptr,
                                            nullptr,
                                            lda,
                                            nullptr,
                                            ldb,
                                            nullptr,
                                            invAsize,
                                            computeType));

        // nullptr for invA => regular trsm
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            N,
                                            alpha,
                                            dA,
                                            lda,
                                            dB,
                                            ldb,
                                            nullptr,
                                            invAsize,
                                            computeType));
    }
}

template <typename T>
void testing_trsm_ex(const Arguments& arg)
{
    bool FORTRAN         = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasTrsmExFn = FORTRAN ? hipblasTrsmExFortran : hipblasTrsmEx;

    hipblasSideMode_t  side   = char2hipblas_side(arg.side);
    hipblasFillMode_t  uplo   = char2hipblas_fill(arg.uplo);
    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(arg.diag);
    int                M      = arg.M;
    int                N      = arg.N;
    int                lda    = arg.lda;
    int                ldb    = arg.ldb;

    T h_alpha = arg.get_alpha<T>();

    int K = (side == HIPBLAS_SIDE_LEFT ? M : N);

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < K || ldb < M)
    {
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hA), `d` is in GPU (device) memory (eg dA).
    // Allocate host memory
    host_matrix<T> hA(K, K, lda);
    host_matrix<T> hB_host(K, K, ldb);
    host_matrix<T> hB_device(M, N, ldb);
    host_matrix<T> hB_cpu(M, N, ldb);

    // Allocate device memory
    device_matrix<T> dA(K, K, lda);
    device_matrix<T> dB(M, N, ldb);
    device_matrix<T> dinvA(TRSM_BLOCK, TRSM_BLOCK, K);
    device_vector<T> d_alpha(1);

    // Check device memory allocation
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dinvA.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial data on CPU
    hipblas_init_matrix(
        hA, arg, hipblas_client_never_set_nan, hipblas_diagonally_dominant_triangular_matrix, true);
    hipblas_init_matrix(hB_host, arg, hipblas_client_never_set_nan, hipblas_general_matrix);

    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, lda, K);
    }

    // Calculate hB = hA*hX;
    ref_trmm<T>(side,
                uplo,
                transA,
                diag,
                M,
                N,
                T(1.0) / h_alpha,
                (const T*)hA.data(),
                lda,
                hB_host.data(),
                ldb);

    hB_cpu = hB_device = hB_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    hipblasStride stride_A    = TRSM_BLOCK * size_t(lda) + TRSM_BLOCK;
    hipblasStride stride_invA = TRSM_BLOCK * TRSM_BLOCK;
    int           blocks      = K / TRSM_BLOCK;

    // Calculate invA
    if(blocks > 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                          uplo,
                                                          diag,
                                                          TRSM_BLOCK,
                                                          dA,
                                                          lda,
                                                          stride_A,
                                                          dinvA,
                                                          TRSM_BLOCK,
                                                          stride_invA,
                                                          blocks));
    }

    if(K % TRSM_BLOCK != 0 || blocks == 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasTrtriStridedBatched<T>(handle,
                                                          uplo,
                                                          diag,
                                                          K - TRSM_BLOCK * blocks,
                                                          dA + stride_A * blocks,
                                                          lda,
                                                          stride_A,
                                                          dinvA + stride_invA * blocks,
                                                          TRSM_BLOCK,
                                                          stride_invA,
                                                          1));
    }

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            N,
                                            &h_alpha,
                                            dA,
                                            lda,
                                            dB,
                                            ldb,
                                            dinvA,
                                            TRSM_BLOCK * K,
                                            arg.compute_type));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hB_host.transfer_from(dB));
        CHECK_HIP_ERROR(dB.transfer_from(hB_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                            side,
                                            uplo,
                                            transA,
                                            diag,
                                            M,
                                            N,
                                            d_alpha,
                                            dA,
                                            lda,
                                            dB,
                                            ldb,
                                            dinvA,
                                            TRSM_BLOCK * K,
                                            arg.compute_type));

        CHECK_HIP_ERROR(hB_device.transfer_from(dB));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_trsm<T>(
            side, uplo, transA, diag, M, N, h_alpha, (const T*)hA.data(), lda, hB_cpu.data(), ldb);

        // if enable norm check, norm check is invasive
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        hipblas_error_host   = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_host);
        hipblas_error_device = norm_check_general<T>('F', M, N, ldb, hB_cpu, hB_device);

        if(arg.unit_check)
        {
            unit_check_error(hipblas_error_host, tolerance);
            unit_check_error(hipblas_error_device, tolerance);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrsmExFn(handle,
                                                side,
                                                uplo,
                                                transA,
                                                diag,
                                                M,
                                                N,
                                                d_alpha,
                                                dA,
                                                lda,
                                                dB,
                                                ldb,
                                                dinvA,
                                                TRSM_BLOCK * K,
                                                arg.compute_type));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasTrsmExModel{}.log_args<T>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         trsm_gflop_count<T>(M, N, K),
                                         trsm_gbyte_count<T>(M, N, K),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }
}
