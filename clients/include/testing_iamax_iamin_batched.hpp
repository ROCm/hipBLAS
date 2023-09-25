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

using hipblasIamaxIaminBatchedModel = ArgumentModel<e_a_type, e_N, e_incx, e_batch_count, e_api>;

template <typename T>
using hipblas_iamax_iamin_batched_t = hipblasStatus_t (*)(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
void testing_iamax_iamin_batched(const Arguments& arg, hipblas_iamax_iamin_batched_t<T> func)
{
    int N           = arg.N;
    int incx        = arg.incx;
    int batch_count = arg.batch_count;

    hipblasLocalHandle handle(arg);
    int                zero = 0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0 || N <= 0 || incx <= 0)
    {
        // quick return success
        device_vector<int> d_hipblas_result_0(std::max(1, batch_count));
        host_vector<int>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        ASSERT_HIP_SUCCESS(hipMemcpy(d_hipblas_result_0,
                                     h_hipblas_result_0,
                                     sizeof(int) * std::max(1, batch_count),
                                     hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(func(handle, N, nullptr, incx, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<int> cpu_0(batch_count);
            host_vector<int> gpu_0(batch_count);
            ASSERT_HIP_SUCCESS(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(int) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<int>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return;
    }

    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<int>     cpu_result(batch_count);
    host_vector<int>     hipblas_result_host(batch_count);
    host_vector<int>     hipblas_result_device(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<int>     d_hipblas_result_device(batch_count);
    ASSERT_HIP_SUCCESS(dx.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, true);
    ASSERT_HIP_SUCCESS(dx.transfer_from(hx));

    double gpu_time_used;
    int    hipblas_error_host = 0, hipblas_error_device = 0;

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        ASSERT_HIP_SUCCESS(hipMemcpy(hipblas_result_device,
                                     d_hipblas_result_device,
                                     sizeof(int) * batch_count,
                                     hipMemcpyDeviceToHost));

        // host_pointer
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            REFBLAS_FUNC(N, hx[b], incx, cpu_result + b);
            // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
            cpu_result[b] += 1;
        }

        if(arg.unit_check)
        {
            unit_check_general<int>(1, 1, batch_count, cpu_result, hipblas_result_host);
            unit_check_general<int>(1, 1, batch_count, cpu_result, hipblas_result_device);
        }
        if(arg.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                hipblas_error_host   = std::max(hipblas_error_host,
                                              std::abs(hipblas_result_host[b] - cpu_result[b]));
                hipblas_error_device = std::max(hipblas_error_device,
                                                std::abs(hipblas_result_device[b] - cpu_result[b]));
            }
        }
    } // end of if unit/norm check

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(
                func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasIamaxIaminBatchedModel{}.log_args<T>(std::cout,
                                                    arg,
                                                    gpu_time_used,
                                                    iamax_gflop_count<T>(N),
                                                    iamax_gbyte_count<T>(N),
                                                    hipblas_error_host,
                                                    hipblas_error_device);
    }
}

inline void testname_iamax_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamax_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasIamaxBatchedFn
        = FORTRAN ? hipblasIamaxBatched<T, true> : hipblasIamaxBatched<T, false>;

    testing_iamax_iamin_batched<T, cblas_iamax<T>>(arg, hipblasIamaxBatchedFn);
}

template <typename T>
hipblasStatus_t testing_iamax_batched_ret(const Arguments& arg)
{
    testing_iamax_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}

inline void testname_iamin_batched(const Arguments& arg, std::string& name)
{
    hipblasIamaxIaminBatchedModel{}.test_name(arg, name);
}

template <typename T>
void testing_iamin_batched(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasIaminBatchedFn
        = FORTRAN ? hipblasIaminBatched<T, true> : hipblasIaminBatched<T, false>;

    testing_iamax_iamin_batched<T, cblas_iamin<T>>(arg, hipblasIaminBatchedFn);
}

template <typename T>
hipblasStatus_t testing_iamin_batched_ret(const Arguments& arg)
{
    testing_iamin_batched<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
