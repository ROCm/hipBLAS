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

/* ============================================================================================ */

using hipblasHerStridedBatchedModel
    = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_lda, e_stride_scale, e_batch_count>;

inline void testname_her_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasHerStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_her_strided_batched_bad_arg(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerStridedBatchedFn
        = FORTRAN ? hipblasHerStridedBatched<T, U, true> : hipblasHerStridedBatched<T, U, false>;

    auto hipblasHerStridedBatchedFn_64 = arg.api == FORTRAN_64
                                             ? hipblasHerStridedBatched_64<T, U, true>
                                             : hipblasHerStridedBatched_64<T, U, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N           = 100;
        int64_t           lda         = 100;
        int64_t           incx        = 1;
        int64_t           batch_count = 2;
        hipblasStride     strideA     = N * lda;
        hipblasStride     stridex     = N * incx;

        device_vector<U> d_alpha(1), d_zero(1);

        const U  h_alpha(1), h_zero(0);
        const U* alpha = &h_alpha;
        const U* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        device_vector<T> dA(strideA * batch_count);
        device_vector<T> dx(stridex * batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHerStridedBatchedFn,
                    (nullptr, uplo, N, alpha, dx, incx, stridex, dA, lda, strideA, batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHerStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     alpha,
                     dx,
                     incx,
                     stridex,
                     dA,
                     lda,
                     strideA,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHerStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     alpha,
                     dx,
                     incx,
                     stridex,
                     dA,
                     lda,
                     strideA,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHerStridedBatchedFn,
                    (handle, uplo, N, nullptr, dx, incx, stridex, dA, lda, strideA, batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {

            // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasHerStridedBatchedFn,
                (handle, uplo, N, alpha, nullptr, incx, stridex, dA, lda, strideA, batch_count));

            DAPI_EXPECT(
                HIPBLAS_STATUS_INVALID_VALUE,
                hipblasHerStridedBatchedFn,
                (handle, uplo, N, alpha, dx, incx, stridex, nullptr, lda, strideA, batch_count));
        }

        // With N == 0, can have all nullptrs
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasHerStridedBatchedFn,
            (handle, uplo, N, alpha, dx, incx, stridex, nullptr, lda, strideA, batch_count));

        // With batch_count == 0, can have all nullptrs
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHerStridedBatchedFn,
                    (handle, uplo, N, nullptr, nullptr, incx, stridex, nullptr, lda, strideA, 0));

        // With alpha == 0, can have all nullptrs
        DAPI_EXPECT(
            HIPBLAS_STATUS_INVALID_VALUE,
            hipblasHerStridedBatchedFn,
            (handle, uplo, N, zero, nullptr, incx, stridex, nullptr, lda, strideA, batch_count));
    }
}

template <typename T>
void testing_her_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHerStridedBatchedFn
        = FORTRAN ? hipblasHerStridedBatched<T, U, true> : hipblasHerStridedBatched<T, U, false>;

    auto hipblasHerStridedBatchedFn_64 = arg.api == FORTRAN_64
                                             ? hipblasHerStridedBatched_64<T, U, true>
                                             : hipblasHerStridedBatched_64<T, U, false>;

    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    int64_t           N            = arg.N;
    int64_t           incx         = arg.incx;
    int64_t           lda          = arg.lda;
    double            stride_scale = arg.stride_scale;
    int64_t           batch_count  = arg.batch_count;

    int64_t abs_incx = incx >= 0 ? incx : -incx;

    hipblasStride stride_A = lda * N * stride_scale;
    hipblasStride stride_x = N * abs_incx * stride_scale;
    size_t        A_size   = stride_A * batch_count;
    size_t        x_size   = stride_x * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || lda < N || lda < 1 || !incx || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHerStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     lda,
                     stride_A,
                     batch_count));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hA_host(A_size);
    host_vector<T> hA_device(A_size);
    host_vector<T> hx(x_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<U> d_alpha(1);

    double hipblas_error_host, hipblas_error_device;

    U h_alpha = arg.get_alpha<U>();

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, N, N, lda, stride_A, batch_count, hipblas_client_never_set_nan, true);
    hipblas_init_vector(
        hx, arg, N, abs_incx, stride_x, batch_count, hipblas_client_alpha_sets_nan, false, true);

    // copy matrix is easy in STL; hA_cpu = hA: save a copy in hA_cpu which will be output of CPU BLAS
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(
            hipblasHerStridedBatchedFn,
            (handle, uplo, N, (U*)&h_alpha, dx, incx, stride_x, dA, lda, stride_A, batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_host.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHerStridedBatchedFn,
                   (handle, uplo, N, d_alpha, dx, incx, stride_x, dA, lda, stride_A, batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_device.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_her<T>(uplo,
                       N,
                       h_alpha,
                       hx.data() + b * stride_x,
                       incx,
                       hA_cpu.data() + b * stride_A,
                       lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, lda, stride_A, hA_cpu.data(), hA_host.data());
            unit_check_general<T>(
                N, N, batch_count, lda, stride_A, hA_cpu.data(), hA_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', N, N, lda, stride_A, hA_cpu.data(), hA_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', N, N, lda, stride_A, hA_cpu.data(), hA_device.data(), batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(
                hipblasHerStridedBatchedFn,
                (handle, uplo, N, d_alpha, dx, incx, stride_x, dA, lda, stride_A, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHerStridedBatchedModel{}.log_args<U>(std::cout,
                                                    arg,
                                                    gpu_time_used,
                                                    her_gflop_count<T>(N),
                                                    her_gbyte_count<T>(N),
                                                    hipblas_error_host,
                                                    hipblas_error_device);
    }
}
