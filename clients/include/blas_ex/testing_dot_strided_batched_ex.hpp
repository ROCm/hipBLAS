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

using hipblasDotStridedBatchedExModel = ArgumentModel<e_a_type,
                                                      e_b_type,
                                                      e_c_type,
                                                      e_compute_type,
                                                      e_N,
                                                      e_incx,
                                                      e_incy,
                                                      e_stride_scale,
                                                      e_batch_count>;

inline void testname_dot_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotStridedBatchedExModel{}.test_name(arg, name);
}

inline void testname_dotc_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotStridedBatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
void testing_dot_strided_batched_ex(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasDotStridedBatchedExFn
        = FORTRAN ? (CONJ ? hipblasDotcStridedBatchedExFortran : hipblasDotStridedBatchedExFortran)
                  : (CONJ ? hipblasDotcStridedBatchedEx : hipblasDotStridedBatchedEx);

    int    N            = arg.N;
    int    incx         = arg.incx;
    int    incy         = arg.incy;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    int           abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = size_t(N) * abs_incx * stride_scale;
    hipblasStride stridey  = size_t(N) * abs_incy * stride_scale;
    size_t        sizeX    = stridex * batch_count;
    size_t        sizeY    = stridey * batch_count;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    hipblasLocalHandle handle(arg);

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t resultType    = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<Tr>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        ASSERT_HIP_SUCCESS(hipMemcpy(d_hipblas_result_0,
                                     h_hipblas_result_0,
                                     sizeof(Tr) * std::max(1, batch_count),
                                     hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasDotStridedBatchedExFn(handle,
                                                            N,
                                                            nullptr,
                                                            xType,
                                                            incx,
                                                            stridex,
                                                            nullptr,
                                                            yType,
                                                            incy,
                                                            stridey,
                                                            batch_count,
                                                            d_hipblas_result_0,
                                                            resultType,
                                                            executionType));

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            ASSERT_HIP_SUCCESS(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy(sizeY);
    host_vector<Tr> h_hipblas_result_host(batch_count);
    host_vector<Tr> h_hipblas_result_device(batch_count);
    host_vector<Tr> h_cpu_result(batch_count);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Tr> d_hipblas_result(batch_count);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(
        hx, arg, N, abs_incx, stridex, batch_count, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(
        hy, arg, N, abs_incy, stridey, batch_count, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device, does not work for incx != 1
    ASSERT_HIP_SUCCESS(hipMemcpy(dx, hx, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dy, hy, sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasDotStridedBatchedExFn(handle,
                                                            N,
                                                            dx,
                                                            xType,
                                                            incx,
                                                            stridex,
                                                            dy,
                                                            yType,
                                                            incy,
                                                            stridey,
                                                            batch_count,
                                                            h_hipblas_result_host,
                                                            resultType,
                                                            executionType));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(hipblasDotStridedBatchedExFn(handle,
                                                            N,
                                                            dx,
                                                            xType,
                                                            incx,
                                                            stridex,
                                                            dy,
                                                            yType,
                                                            incy,
                                                            stridey,
                                                            batch_count,
                                                            d_hipblas_result,
                                                            resultType,
                                                            executionType));

        ASSERT_HIP_SUCCESS(hipMemcpy(h_hipblas_result_device,
                                     d_hipblas_result,
                                     sizeof(Tr) * batch_count,
                                     hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<Tx> : cblas_dot<Tx>)(N,
                                                    hx.data() + b * stridex,
                                                    incx,
                                                    hy.data() + b * stridey,
                                                    incy,
                                                    &h_cpu_result[b]);
        }

        if(arg.unit_check)
        {
            if(std::is_same<Tr, hipblasHalf>{})
            {
                double tol = error_tolerance<Tr> * N;
                near_check_general(1,
                                   1,
                                   batch_count,
                                   1,
                                   1,
                                   h_cpu_result.data(),
                                   h_hipblas_result_host.data(),
                                   tol);
                near_check_general(1,
                                   1,
                                   batch_count,
                                   1,
                                   1,
                                   h_cpu_result.data(),
                                   h_hipblas_result_device.data(),
                                   tol);
            }
            else
            {
                unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
                unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
            hipblas_error_device = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
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

            ASSERT_HIPBLAS_SUCCESS(hipblasDotStridedBatchedExFn(handle,
                                                                N,
                                                                dx,
                                                                xType,
                                                                incx,
                                                                stridex,
                                                                dy,
                                                                yType,
                                                                incy,
                                                                stridey,
                                                                batch_count,
                                                                d_hipblas_result,
                                                                resultType,
                                                                executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotStridedBatchedExModel{}.log_args<Tx>(std::cout,
                                                       arg,
                                                       gpu_time_used,
                                                       dot_gflop_count<CONJ, Tx>(N),
                                                       dot_gbyte_count<Tx>(N),
                                                       hipblas_error_host,
                                                       hipblas_error_device);
    }
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr>
void testing_dotc_strided_batched_ex(const Arguments& arg)
{
    testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, true>(arg);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
hipblasStatus_t testing_dot_strided_batched_ex_ret(const Arguments& arg)
{
    testing_dot_strided_batched_ex<Tx, Ty, Tr, Tex, CONJ>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}