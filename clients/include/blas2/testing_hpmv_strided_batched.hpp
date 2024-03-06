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

using hipblasHpmvStridedBatchedModel = ArgumentModel<e_a_type,
                                                     e_uplo,
                                                     e_N,
                                                     e_alpha,
                                                     e_incx,
                                                     e_beta,
                                                     e_incy,
                                                     e_stride_scale,
                                                     e_batch_count>;

inline void testname_hpmv_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasHpmvStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_hpmv_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHpmvStridedBatchedFn
        = FORTRAN ? hipblasHpmvStridedBatched<T, true> : hipblasHpmvStridedBatched<T, false>;

    auto hipblasHpmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHpmvStridedBatched_64<T, true>
                                              : hipblasHpmvStridedBatched_64<T, false>;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        hipblasFillMode_t uplo        = HIPBLAS_FILL_MODE_UPPER;
        int64_t           N           = 100;
        int64_t           incx        = 1;
        int64_t           incy        = 1;
        int64_t           batch_count = 2;
        int64_t           A_size      = N * (N + 1) / 2;
        hipblasStride     strideA     = A_size;
        hipblasStride     stridex     = N * incx;
        hipblasStride     stridey     = N * incy;

        device_vector<T> d_alpha(1), d_beta(1), d_one(1), d_zero(1);

        const T  h_alpha(1), h_beta(2), h_one(1), h_zero(0);
        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;
        const T* one   = &h_one;
        const T* zero  = &h_zero;

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(*beta), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_one, one, sizeof(*one), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            beta  = d_beta;
            one   = d_one;
            zero  = d_zero;
        }

        device_vector<T> dA(strideA * batch_count);
        device_vector<T> dx(stridex * batch_count);
        device_vector<T> dy(stridey * batch_count);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasHpmvStridedBatchedFn,
                    (nullptr,
                     uplo,
                     N,
                     alpha,
                     dA,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUEhipblasHpmvStridedBatchedFn,
                    (handle,
                     HIPBLAS_FILL_MODE_FULL,
                     N,
                     alpha,
                     dA,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasHpmvStridedBatchedFn,
                    (handle,
                     (hipblasFillMode_t)HIPBLAS_OP_N,
                     N,
                     alpha,
                     dA,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHpmvStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     nullptr,
                     dA,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     beta,
                     dy,
                     incy,
                     stridey,
                     batch_count));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasHpmvStridedBatchedFn,
                    (handle,
                     uplo,
                     N,
                     alpha,
                     dA,
                     strideA,
                     dx,
                     incx,
                     stridex,
                     nullptr,
                     dy,
                     incy,
                     stridey,
                     batch_count));

        if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
        {
            // For device mode in rocBLAS we don't have checks for dA, dx, dy as we may be able to quick return
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHpmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         nullptr,
                         strideA,
                         dx,
                         incx,
                         stridex,
                         beta,
                         dy,
                         incy,
                         stridey,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUEhipblasHpmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
                         strideA,
                         nullptr,
                         incx,
                         stridex,
                         beta,
                         dy,
                         incy,
                         stridey,
                         batch_count));
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasHpmvStridedBatchedFn,
                        (handle,
                         uplo,
                         N,
                         alpha,
                         dA,
                         strideA,
                         dx,
                         incx,
                         stridex,
                         beta,
                         nullptr,
                         incy,
                         stridey,
                         batch_count));
        }

        // Check 64-bit API with quick return
        if(arg.api & c_API_64)
        {
            DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                       (handle,
                        uplo,
                        N,
                        zero,
                        dA,
                        strideA,
                        dx,
                        incx,
                        stridex,
                        one,
                        dy,
                        incy,
                        stridey,
                        batch_count));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    0,
                    nullptr,
                    nullptr,
                    strideA,
                    nullptr,
                    incx,
                    stridex,
                    nullptr,
                    nullptr,
                    incy,
                    stridey,
                    batch_count));
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    nullptr,
                    nullptr,
                    strideA,
                    nullptr,
                    incx,
                    stridex,
                    nullptr,
                    nullptr,
                    incy,
                    stridey,
                    0));

        // With alpha == 0 can have A and x nullptr
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    zero,
                    nullptr,
                    strideA,
                    nullptr,
                    incx,
                    stridex,
                    beta,
                    dy,
                    incy,
                    stridey,
                    batch_count));

        // With alpha == 0 && beta == 1, all other ptrs can be nullptr
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    zero,
                    nullptr,
                    strideA,
                    nullptr,
                    incx,
                    stridex,
                    one,
                    nullptr,
                    incy,
                    stridey,
                    batch_count));
    }
}

template <typename T>
void testing_hpmv_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasHpmvStridedBatchedFn
        = FORTRAN ? hipblasHpmvStridedBatched<T, true> : hipblasHpmvStridedBatched<T, false>;

    auto hipblasHpmvStridedBatchedFn_64 = arg.api == FORTRAN_64
                                              ? hipblasHpmvStridedBatched_64<T, true>
                                              : hipblasHpmvStridedBatched_64<T, false>;

    hipblasFillMode_t uplo         = char2hipblas_fill(arg.uplo);
    int64_t           N            = arg.N;
    int64_t           incx         = arg.incx;
    int64_t           incy         = arg.incy;
    double            stride_scale = arg.stride_scale;
    int64_t           batch_count  = arg.batch_count;

    size_t        abs_incx = incx >= 0 ? incx : -incx;
    size_t        abs_incy = incy >= 0 ? incy : -incy;
    size_t        dim_A    = N * (N + 1) / 2;
    hipblasStride stride_A = dim_A * stride_scale;
    hipblasStride stride_x = N * abs_incx * stride_scale;
    hipblasStride stride_y = N * abs_incy * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t X_size = stride_x * batch_count;
    size_t Y_size = stride_y * batch_count;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasHpmvStridedBatchedFn(handle,
                                                uplo,
                                                N,
                                                nullptr,
                                                nullptr,
                                                stride_A,
                                                nullptr,
                                                incx,
                                                stride_x,
                                                nullptr,
                                                nullptr,
                                                incy,
                                                stride_y,
                                                batch_count));
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hy_cpu(Y_size);
    host_vector<T> hy_host(Y_size);
    host_vector<T> hy_device(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double hipblas_error_host, hipblas_error_device;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, dim_A, 1, 1, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(
        hx, arg, N, abs_incx, stride_x, batch_count, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, N, abs_incy, stride_y, batch_count, hipblas_client_beta_sets_nan);

    // copy vector is easy in STL; hy_cpu = hy: save a copy in hy_cpu which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    (T*)&h_alpha,
                    dA,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    (T*)&h_beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasHpmvStridedBatchedFn,
                   (handle,
                    uplo,
                    N,
                    d_alpha,
                    dA,
                    stride_A,
                    dx,
                    incx,
                    stride_x,
                    d_beta,
                    dy,
                    incy,
                    stride_y,
                    batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hy_device.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(size_t b = 0; b < batch_count; b++)
        {
            ref_hpmv<T>(uplo,
                        N,
                        h_alpha,
                        hA.data() + b * stride_A,
                        hx.data() + b * stride_x,
                        incx,
                        h_beta,
                        hy_cpu.data() + b * stride_y,
                        incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_cpu, hy_host);
            unit_check_general<T>(1, N, batch_count, abs_incy, stride_y, hy_cpu, hy_device);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu, hy_host, batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, N, abs_incy, stride_y, hy_cpu, hy_device, batch_count);
        }
    }

    if(arg.timing)
    {
        double gpu_time_used;
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasHpmvStridedBatchedFn,
                          (handle,
                           uplo,
                           N,
                           d_alpha,
                           dA,
                           stride_A,
                           dx,
                           incx,
                           stride_x,
                           d_beta,
                           dy,
                           incy,
                           stride_y,
                           batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasHpmvStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     hpmv_gflop_count<T>(N),
                                                     hpmv_gbyte_count<T>(N),
                                                     hipblas_error_host,
                                                     hipblas_error_device);
    }
}
