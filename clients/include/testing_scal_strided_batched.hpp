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

using hipblasScalStridedBatchedModel
    = ArgumentModel<e_a_type, e_c_type, e_N, e_alpha, e_incx, e_stride_scale, e_batch_count, e_api>;

inline void testname_scal_strided_batched(const Arguments& arg, std::string& name)
{
    hipblasScalStridedBatchedModel{}.test_name(arg, name);
}

template <typename T, typename U = T>
void testing_scal_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasScalStridedBatchedFn
        = FORTRAN ? hipblasScalStridedBatched<T, U, true> : hipblasScalStridedBatched<T, U, false>;

    int    N            = arg.N;
    int    incx         = arg.incx;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    U alpha = arg.get_alpha<U>();

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        ASSERT_HIPBLAS_SUCCESS(
            hipblasScalStridedBatchedFn(handle, N, nullptr, nullptr, incx, stridex, batch_count));
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hx(sizeX);
    host_vector<T> hz(sizeX);

    device_vector<T> dx(sizeX);

    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, arg, N, incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    // copy data from CPU to device, does not work for incx != 1
    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx.data(), sizeof(T) * sizeX, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(
            hipblasScalStridedBatchedFn(handle, N, &alpha, dx, incx, stridex, batch_count));

        // copy output from device to CPU
        ASSERT_HIP_SUCCESS(hipMemcpy(hx.data(), dx, sizeof(T) * sizeX, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<T, U>(N, alpha, hz.data() + b * stridex, incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, stridex, hz, hx);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(
                hipblasScalStridedBatchedFn(handle, N, &alpha, dx, incx, stridex, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasScalStridedBatchedModel{}.log_args<T>(std::cout,
                                                     arg,
                                                     gpu_time_used,
                                                     scal_gflop_count<T, U>(N),
                                                     scal_gbyte_count<T>(N),
                                                     hipblas_error);
    }
}

template <typename T, typename U = T>
hipblasStatus_t testing_scal_strided_batched_ret(const Arguments& arg)
{
    testing_scal_strided_batched<T, U>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
