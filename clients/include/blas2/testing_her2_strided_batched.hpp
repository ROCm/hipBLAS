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

using hipblasHer2StridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_N,
                                                     e_alpha,
                                                     e_incx,
                                                     e_incy,
                                                     e_lda,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_her2_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasHer2StridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_her2_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHer2StridedBatchedFn
        = FORTRAN ? hipblasHer2StridedBatched<T, true> : hipblasHer2StridedBatched<T, false>;

    auto hipblasHer2StridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHer2StridedBatched_64<T, true>
                                              : hipblasHer2StridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N           = 100;
        int64_t           lda         = 100;
        int64_t           incx        = 1;
        int64_t           incy        = 1;
        int64_t           batch_count = 2;
        hipblasStride     stride_x    = N * incx;
        hipblasStride     stride_y    = N * incy;
        hipblasStride     stride_A    = N * lda;

        device_vector<T> d_alpha(1), d_zero(1);

        const T  h_alpha(1), h_zero(0);
        const T* alpha = &h_alpha;
        const T* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
        device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
        device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHer2StridedBatchedFn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dx,
                     incx,
                     stride_x,
                     dy,
                     incy,
                     stride_y,
                     dA,
                     lda,
                     stride_A,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHer2StridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     alpha,
                     dx,
                     incx,
                     stride_x,
                     dy,
                     incy,
                     stride_y,
                     dA,
                     lda,
                     stride_A,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHer2StridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     alpha,
                     dx,
                     incx,
                     stride_x,
                     dy,
                     incy,
                     stride_y,
                     dA,
                     lda,
                     stride_A,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHer2StridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dx,
                     incx,
                     stride_x,
                     dy,
                     incy,
                     stride_y,
                     dA,
                     lda,
                     stride_A,
                     batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dA, dx as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHer2StridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         nullptr,
                         incx,
                         stride_x,
                         dy,
                         incy,
                         stride_y,
                         dA,
                         lda,
                         stride_A,
                         batch_count));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHer2StridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dx,
                         incx,
                         stride_x,
                         nullptr,
                         incy,
                         stride_y,
                         dA,
                         lda,
                         stride_A,
                         batch_count));

            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHer2StridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dx,
                         incx,
                         stride_x,
                         dy,
                         incy,
                         stride_y,
                         nullptr,
                         lda,
                         stride_A,
                         batch_count));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasHer2StridedBatchedFn,
                   (handle,
                    uplo,
                    0,
                    nullptr,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    lda,
                    stride_A,
                    batch_count));

        // With batch_count == 0, can have all nullptrs
        DAPI_CHECK(hipblasHer2StridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    nullptr,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    lda,
                    stride_A,
                    0));

        // With alpha == 0, can have all nullptrs
        DAPI_CHECK(hipblasHer2StridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    zero,
                    nullptr,
                    incx,
                    stride_x,
                    nullptr,
                    incy,
                    stride_y,
                    nullptr,
                    lda,
                    stride_A,
                    batch_count));
    }
}

template <typename T>
void testing_her2_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHer2StridedBatchedFn
        = FORTRAN ? hipblasHer2StridedBatched<T, true> : hipblasHer2StridedBatched<T, false>;

    auto hipblasHer2StridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHer2StridedBatched_64<T, true>
                                              : hipblasHer2StridedBatched_64<T, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    int64_t lda          = arg.lda;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int64_t abs_incx = incx >= 0 ? incx : -incx;
    int64_t abs_incy = incy >= 0 ? incy : -incy;

    hipblasStride     stride_A = size_t(lda) * N * stride_scale;
    hipblasStride     stride_x = size_t(N) * abs_incx * stride_scale;
    hipblasStride     stride_y = size_t(N) * abs_incy * stride_scale;
    hipblasFillMode_t uplo     = char2hipblas_fill(arg.uplo);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || lda < N || lda < 1 || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHer2StridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     nullptr,
                     incx,
                     stride_x,
                     nullptr,
                     incy,
                     stride_y,
                     nullptr,
                     lda,
                     stride_A,
                     batch_count));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_strided_batch_matrix<T> hA(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hA_cpu(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hA_host(N, N, lda, stride_A, batch_count);
    host_strided_batch_matrix<T> hA_device(N, N, lda, stride_A, batch_count);
    host_strided_batch_vector<T> hx(N, incx, stride_x, batch_count);
    host_strided_batch_vector<T> hy(N, incy, stride_y, batch_count);

    device_strided_batch_matrix<T> dA(N, N, lda, stride_A, batch_count);
    device_strided_batch_vector<T> dx(N, incx, stride_x, batch_count);
    device_strided_batch_vector<T> dy(N, incy, stride_y, batch_count);
    device_vector<T>               d_alpha(1);

    double hipblas_error_host, hipblas_error_device;

    T h_alpha = arg.get_alpha<T>();

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_hermitian_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan);

    // copy matrix is easy in STL; hA_cpu = hA: save a copy in hA_cpu which will be output of CPU BLAS
    hA_cpu.copy_from(hA);

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHer2StridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    (T*)&h_alpha,
                    dx,
                    incx,
                    stride_x,
                    dy,
                    incy,
                    stride_y,
                    dA,
                    lda,
                    stride_A,
                    batch_count));

        CHECK_HIP_ERROR(hA_host.transfer_from(dA));
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHer2StridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    d_alpha,
                    dx,
                    incx,
                    stride_x,
                    dy,
                    incy,
                    stride_y,
                    dA,
                    lda,
                    stride_A,
                    batch_count));

        CHECK_HIP_ERROR(hA_device.transfer_from(dA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t b = 0; b < batch_count; b++)
        {
            ref_her2<T>(uplo, N, h_alpha, hx[b], incx, hy[b], incy, hA_cpu[b], lda);
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
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasHer2StridedBatchedFn,
                          (handle,
                           uplo,
                           N,
                           d_alpha,
                           dx,
                           incx,
                           stride_x,
                           dy,
                           incy,
                           stride_y,
                           dA,
                           lda,
                           stride_A,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHer2StridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     her2_gflop_count<T>(N),
                                                     her2_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
