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

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasSetGetVectorAsyncModel = ArgumentModel<e_a_type, e_M, e_incx, e_incy, e_incd>;

inline void testname_set_get_vector_async(const Arguments& arg, std::string& name)
{
    hipblasSetGetVectorAsyncModel{}.test_name(arg, name);
}

template <typename T>
void testing_set_get_vector_async(const Arguments& arg)
{
    bool FORTRAN                 = arg.fortran;
    auto hipblasSetVectorAsyncFn = FORTRAN ? hipblasSetVectorAsyncFortran : hipblasSetVectorAsync;
    auto hipblasGetVectorAsyncFn = FORTRAN ? hipblasGetVectorAsyncFortran : hipblasGetVectorAsync;

    int M    = arg.M;
    int incx = arg.incx;
    int incy = arg.incy;
    int incd = arg.incd;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incd <= 0)
    {
        return;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * incx);
    host_vector<T> hy(M * incy);
    host_vector<T> hy_ref(M * incy);

    device_vector<T> db(M * incd);

    double             hipblas_error = 0.0, gpu_time_used = 0.0;
    hipblasLocalHandle handle(arg);

    hipStream_t stream;
    hipblasGetStream(handle, &stream);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, M, incy);
    hy_ref = hy;

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    ASSERT_HIPBLAS_SUCCESS(
        hipblasSetVectorAsyncFn(M, sizeof(T), (void*)hx, incx, (void*)db, incd, stream));
    ASSERT_HIPBLAS_SUCCESS(
        hipblasGetVectorAsyncFn(M, sizeof(T), (void*)db, incd, (void*)hy, incy, stream));

    ASSERT_HIP_SUCCESS(hipStreamSynchronize(stream));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_ref[i * incy] = hx[i * incx];
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy.data(), hy_ref.data());
        }
        if(arg.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, incy, hy, hy_ref);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            ASSERT_HIPBLAS_SUCCESS(
                hipblasSetVectorAsyncFn(M, sizeof(T), (void*)hx, incx, (void*)db, incd, stream));
            ASSERT_HIPBLAS_SUCCESS(
                hipblasGetVectorAsyncFn(M, sizeof(T), (void*)db, incd, (void*)hy, incy, stream));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSetGetVectorAsyncModel{}.log_args<T>(std::cout,
                                                    arg,
                                                    gpu_time_used,
                                                    ArgumentLogging::NA_value,
                                                    set_get_vector_gbyte_count<T>(M),
                                                    hipblas_error);
    }
}

template <typename T>
hipblasStatus_t testing_set_get_vector_async_ret(const Arguments& arg)
{
    testing_set_get_vector_async<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}