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

using hipblasDotBatchedExModel = ArgumentModel<e_N, e_incx, e_incy, e_batch_count>;

inline void testname_dot_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotBatchedExModel{}.test_name(arg, name);
}

inline void testname_dotc_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasDotBatchedExModel{}.test_name(arg, name);
}

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
inline hipblasStatus_t testing_dot_batched_ex_template(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasDotBatchedExFn
        = FORTRAN ? (CONJ ? hipblasDotcBatchedExFortran : hipblasDotBatchedExFortran)
                  : (CONJ ? hipblasDotcBatchedEx : hipblasDotBatchedEx);

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

    hipblasLocalHandle handle(arg);

    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;
    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<Tr>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(d_hipblas_result_0,
                                  h_hipblas_result_0,
                                  sizeof(Tr) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                                  N,
                                                  nullptr,
                                                  xType,
                                                  incx,
                                                  nullptr,
                                                  yType,
                                                  incy,
                                                  batch_count,
                                                  d_hipblas_result_0,
                                                  resultType,
                                                  executionType));

        if(batch_count > 0)
        {
            host_vector<Tr> cpu_0(batch_count);
            host_vector<Tr> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy(N, incy, batch_count);
    host_vector<Tr>       h_cpu_result(batch_count);
    host_vector<Tr>       h_hipblas_result_host(batch_count);
    host_vector<Tr>       h_hipblas_result_device(batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tr>       d_hipblas_result(batch_count);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init(hy, true, false);
    hipblas_init_alternating_sign(hx);
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                                  N,
                                                  dx.ptr_on_device(),
                                                  xType,
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  yType,
                                                  incy,
                                                  batch_count,
                                                  h_hipblas_result_host,
                                                  resultType,
                                                  executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                                  N,
                                                  dx.ptr_on_device(),
                                                  xType,
                                                  incx,
                                                  dy.ptr_on_device(),
                                                  yType,
                                                  incy,
                                                  batch_count,
                                                  d_hipblas_result,
                                                  resultType,
                                                  executionType));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<Tx>
                  : cblas_dot<Tx>)(N, hx[b], incx, hy[b], incy, &(h_cpu_result[b]));
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
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                                      N,
                                                      dx.ptr_on_device(),
                                                      xType,
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      yType,
                                                      incy,
                                                      batch_count,
                                                      d_hipblas_result,
                                                      resultType,
                                                      executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasDotBatchedExModel{}.log_args<Tx>(std::cout,
                                                arg,
                                                gpu_time_used,
                                                dot_gflop_count<CONJ, Tx>(N),
                                                dot_gbyte_count<Tx>(N),
                                                hipblas_error_host,
                                                hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

inline hipblasStatus_t testing_dot_batched_ex(const Arguments& arg)
{
    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIP_R_16F && yType == HIP_R_16F && resultType == HIP_R_16F
       && executionType == HIP_R_16F)
    {
        status = testing_dot_batched_ex_template<hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 false>(arg);
    }
    else if(xType == HIP_R_16F && yType == HIP_R_16F && resultType == HIP_R_16F
            && executionType == HIP_R_32F)
    {
        status
            = testing_dot_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, false>(
                arg);
    }
    else if(xType == HIP_R_16BF && yType == HIP_R_16BF && resultType == HIP_R_16BF
            && executionType == HIP_R_32F)
    {
        status = testing_dot_batched_ex_template<hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 false>(arg);
    }
    else if(xType == HIP_R_32F && yType == HIP_R_32F && resultType == HIP_R_32F
            && executionType == HIP_R_32F)
    {
        status = testing_dot_batched_ex_template<float, float, float, float, false>(arg);
    }
    else if(xType == HIP_R_64F && yType == HIP_R_64F && resultType == HIP_R_64F
            && executionType == HIP_R_64F)
    {
        status = testing_dot_batched_ex_template<double, double, double, double, false>(arg);
    }
    else if(xType == HIP_C_32F && yType == HIP_C_32F && resultType == HIP_C_32F
            && executionType == HIP_C_32F)
    {
        status = testing_dot_batched_ex_template<hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 false>(arg);
    }
    else if(xType == HIP_C_64F && yType == HIP_C_64F && resultType == HIP_C_64F
            && executionType == HIP_C_64F)
    {
        status = testing_dot_batched_ex_template<hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 false>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

inline hipblasStatus_t testing_dotc_batched_ex(const Arguments& arg)
{
    hipDataType xType         = arg.a_type;
    hipDataType yType         = arg.b_type;
    hipDataType resultType    = arg.c_type;
    hipDataType executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIP_R_16F && yType == HIP_R_16F && resultType == HIP_R_16F
       && executionType == HIP_R_16F)
    {
        status = testing_dot_batched_ex_template<hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 true>(arg);
    }
    else if(xType == HIP_R_16F && yType == HIP_R_16F && resultType == HIP_R_16F
            && executionType == HIP_R_32F)
    {
        status
            = testing_dot_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, true>(
                arg);
    }
    else if(xType == HIP_R_16BF && yType == HIP_R_16BF && resultType == HIP_R_16BF
            && executionType == HIP_R_32F)
    {
        status = testing_dot_batched_ex_template<hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 true>(arg);
    }
    else if(xType == HIP_R_32F && yType == HIP_R_32F && resultType == HIP_R_32F
            && executionType == HIP_R_32F)
    {
        status = testing_dot_batched_ex_template<float, float, float, float, true>(arg);
    }
    else if(xType == HIP_R_64F && yType == HIP_R_64F && resultType == HIP_R_64F
            && executionType == HIP_R_64F)
    {
        status = testing_dot_batched_ex_template<double, double, double, double, true>(arg);
    }
    else if(xType == HIP_C_32F && yType == HIP_C_32F && resultType == HIP_C_32F
            && executionType == HIP_C_32F)
    {
        status = testing_dot_batched_ex_template<hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 true>(arg);
    }
    else if(xType == HIP_C_64F && yType == HIP_C_64F && resultType == HIP_C_64F
            && executionType == HIP_C_64F)
    {
        status = testing_dot_batched_ex_template<hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 hipblasDoubleComplex,
                                                 true>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
