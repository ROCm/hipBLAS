/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasCopyStridedBatchedModel
    = ArgumentModel<e_a_type, e_N, e_incx, e_incy, e_stride_scale, e_batch_count>;

inline void testname_copy_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasCopyStridedBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_copy_strided_batched_bad_arg(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyStridedBatchedFn
        = FORTRAN ? hipblasCopyStridedBatched<T, true> : hipblasCopyStridedBatched<T, false>;
    auto hipblasCopyStridedBatchedFn_64
        = FORTRAN ? hipblasCopyStridedBatched_64<T, true> : hipblasCopyStridedBatched_64<T, false>;

    hipblasLocalHandle handle(arg);

    int64_t       N           = 100;
    int64_t       incx        = 1;
    int64_t       incy        = 1;
    int64_t       batch_count = 2;
    hipblasStride stride_x    = N;
    hipblasStride stride_y    = N;

    device_vector<T> dx(stride_y * batch_count);
    device_vector<T> dy(stride_x * batch_count);

    DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                hipblasCopyStridedBatchedFn,
                (nullptr, N, dx, stride_x, incx, dy, incy, stride_y, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasCopyStridedBatchedFn,
                (handle, N, nullptr, incx, stride_x, dy, incy, stride_y, batch_count));
    DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                hipblasCopyStridedBatchedFn,
                (handle, N, dx, incx, stride_x, nullptr, incy, stride_y, batch_count));
}

template <typename T>
void testing_copy_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasCopyStridedBatchedFn
        = FORTRAN ? hipblasCopyStridedBatched<T, true> : hipblasCopyStridedBatched<T, false>;
    auto hipblasCopyStridedBatchedFn_64
        = FORTRAN ? hipblasCopyStridedBatched_64<T, true> : hipblasCopyStridedBatched_64<T, false>;

    int64_t N            = arg.N;
    int64_t incx         = arg.incx;
    int64_t incy         = arg.incy;
    double  stride_scale = arg.stride_scale;
    int64_t batch_count  = arg.batch_count;

    int64_t       abs_incx = incx >= 0 ? incx : -incx;
    int64_t       abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey  = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX    = stridex * batch_count;
    size_t        sizeY    = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        DAPI_CHECK(hipblasCopyStridedBatchedFn,
                   (handle, N, nullptr, incx, stridex, nullptr, incy, stridey, batch_count));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hy(sizeY);
    host_vector<T> hx_cpu(sizeX);
    host_vector<T> hy_cpu(sizeY);

    device_vector<T> dx(sizeX);
    device_vector<T> dy(sizeY);

    double gpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, arg, N, abs_incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(
        hy, arg, N, abs_incy, stridey, batch_count, hipblas_client_alpha_sets_nan, false);

    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * sizeY, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        DAPI_CHECK(hipblasCopyStridedBatchedFn,
                   (handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * sizeY, hipMemcpyDeviceToHost));

        /*=====================================================================
                    CPU BLAS
        =================================================================== */
        for(int64_t b = 0; b < batch_count; b++)
        {
            cblas_copy<T>((int)N,
                          hx_cpu.data() + b * stridex,
                          (int)incx,
                          hy_cpu.data() + b * stridey,
                          (int)incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy.data());
        }
        if(arg.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, N, abs_incy, stridey, hy_cpu, hy, batch_count);
        }
    } // end of if unit check

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_CHECK(hipblasCopyStridedBatchedFn,
                       (handle, N, dx, incx, stridex, dy, incy, stridey, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasCopyStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     copy_gflop_count<T>(N),
                                                     copy_gbyte_count<T>(N),
                                                     hipblas_error);
    }
}
