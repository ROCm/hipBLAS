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

using hipblasRotmgModel = ArgumentModel<e_a_type>;

inline void testname_rotmg(const Arguments& arg, std::string& name)
{
    hipblasRotmgModel{}.test_name(arg, name);
}

template <typename T>
void testing_rotmg(const Arguments& arg)
{
    bool FORTRAN        = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasRotmgFn = FORTRAN ? hipblasRotmg<T, true> : hipblasRotmg<T, false>;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    host_vector<T> hparams(9);

    const T rel_error = std::numeric_limits<T>::epsilon() * 1000;

    // Initial data on CPU
    hipblas_init_vector(hparams, arg, 9, 1, 0, 1, hipblas_client_alpha_sets_nan, true);

    host_vector<T>   cparams   = hparams;
    host_vector<T>   hparams_d = hparams;
    device_vector<T> dparams(9);
    ASSERT_HIP_SUCCESS(hipMemcpy(dparams, hparams, 9 * sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        ASSERT_HIPBLAS_SUCCESS(hipblasRotmgFn(
            handle, &hparams[0], &hparams[1], &hparams[2], &hparams[3], &hparams[4]));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        ASSERT_HIPBLAS_SUCCESS(
            hipblasRotmgFn(handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));

        ASSERT_HIP_SUCCESS(hipMemcpy(hparams_d, dparams, 9 * sizeof(T), hipMemcpyDeviceToHost));

        // CPU BLAS
        cblas_rotmg<T>(&cparams[0], &cparams[1], &cparams[2], &cparams[3], &cparams[4]);

        if(arg.unit_check)
        {
            near_check_general(1, 9, 1, cparams.data(), hparams.data(), rel_error);
            near_check_general(1, 9, 1, cparams.data(), hparams_d.data(), rel_error);
        }

        if(arg.norm_check)
        {
            hipblas_error_host   = norm_check_general<T>('F', 1, 9, 1, cparams, hparams);
            hipblas_error_device = norm_check_general<T>('F', 1, 9, 1, cparams, hparams_d);
        }
    }

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

            ASSERT_HIPBLAS_SUCCESS(hipblasRotmgFn(
                handle, dparams, dparams + 1, dparams + 2, dparams + 3, dparams + 4));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasRotmgModel{}.log_args<T>(std::cout,
                                        arg,
                                        gpu_time_used,
                                        ArgumentLogging::NA_value,
                                        ArgumentLogging::NA_value,
                                        hipblas_error_host,
                                        hipblas_error_device);
    }
}

template <typename T>
hipblasStatus_t testing_rotmg_ret(const Arguments& arg)
{
    testing_rotmg<T>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
