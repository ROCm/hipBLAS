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

using hipblasSpr2Model = ArgumentModel<e_a_type, e_uplo, e_N, e_alpha, e_incx, e_incy>;

inline void testname_spr2(const Arguments& arg, std::string& name)
{
    hipblasSpr2Model{}.test_name(arg, name);
}

template <typename T>
void testing_spr2_bad_arg(const Arguments& arg)
{
    auto hipblasSpr2Fn = arg.api == FORTRAN ? hipblasSpr2<T, true> : hipblasSpr2<T, false>;
    auto hipblasSpr2Fn_64
        = arg.api == FORTRAN_64 ? hipblasSpr2_64<T, true> : hipblasSpr2_64<T, false>;

    const T           h_alpha(1), h_zero(0);
    const T*          alpha = &h_alpha;
    const T*          zero  = &h_zero;
    hipblasFillMode_t uplo  = HIPBLAS_FILL_MODE_UPPER;

    for(auto pointer_mode : {HIPBLAS_POINTER_MODE_HOST, HIPBLAS_POINTER_MODE_DEVICE})
    {
        hipblasLocalHandle handle(arg);
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, pointer_mode));

        int64_t N    = 100;
        int64_t incx = 1;
        int64_t incy = 1;

        device_vector<T> d_alpha(1), d_zero(1);

        if(pointer_mode == HIPBLAS_POINTER_MODE_DEVICE)
        {
            CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(*alpha), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(d_zero, zero, sizeof(*zero), hipMemcpyHostToDevice));
            alpha = d_alpha;
            zero  = d_zero;
        }

        // Allocate device memory
        device_matrix<T> dAp(1, hipblas_packed_matrix_size(N), 1);
        device_vector<T> dx(N, incx);
        device_vector<T> dy(N, incy);

        DAPI_EXPECT(HIPBLAS_STATUS_NOT_INITIALIZED,
                    hipblasSpr2Fn,
                    (nullptr, uplo, N, alpha, dx, incx, dy, incy, dAp));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                    hipblasSpr2Fn,
                    (handle, HIPBLAS_FILL_MODE_FULL, N, alpha, dx, incx, dy, incy, dAp));
        DAPI_EXPECT(HIPBLAS_STATUS_INVALID_ENUM,
                    hipblasSpr2Fn,
                    (handle, (hipblasFillMode_t)HIPBLAS_OP_N, N, alpha, dx, incx, dy, incy, dAp));

        if(arg.bad_arg_all)
        {
            DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                        hipblasSpr2Fn,
                        (handle, uplo, N, nullptr, dx, incx, dy, incy, dAp));

            if(pointer_mode == HIPBLAS_POINTER_MODE_HOST)
            {
                // For device mode in rocBLAS we don't have checks for dAp, dx as we may be able to quick return
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSpr2Fn,
                            (handle, uplo, N, alpha, nullptr, incx, dy, incy, dAp));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSpr2Fn,
                            (handle, uplo, N, alpha, dx, incx, nullptr, incy, dAp));
                DAPI_EXPECT(HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSpr2Fn,
                            (handle, uplo, N, alpha, dx, incx, dy, incy, nullptr));

                // rocBLAS implementation has alpha == 0 quick return after arg checks, so if we're using 32-bit params,
                // this should fail with invalid-value
                // Note that this strategy can't check incx as rocBLAS supports negative. Also depends on implementation so not testing cuBLAS for now
                DAPI_EXPECT((arg.api & c_API_64) ? HIPBLAS_STATUS_SUCCESS
                                                 : HIPBLAS_STATUS_INVALID_VALUE,
                            hipblasSpr2Fn,
                            (handle, uplo, c_i32_overflow, zero, nullptr, 1, nullptr, 1, nullptr));
            }

            // With alpha == 0, can have all nullptrs
            DAPI_CHECK(hipblasSpr2Fn,
                       (handle, uplo, N, zero, nullptr, incx, nullptr, incy, nullptr));
        }

        // With N == 0, can have all nullptrs
        DAPI_CHECK(hipblasSpr2Fn,
                   (handle, uplo, 0, nullptr, nullptr, incx, nullptr, incy, nullptr));
    }
}

template <typename T>
void testing_spr2(const Arguments& arg)
{
    auto hipblasSpr2Fn = arg.api == FORTRAN ? hipblasSpr2<T, true> : hipblasSpr2<T, false>;
    auto hipblasSpr2Fn_64
        = arg.api == FORTRAN_64 ? hipblasSpr2_64<T, true> : hipblasSpr2_64<T, false>;

    hipblasFillMode_t uplo = char2hipblas_fill(arg.uplo);
    int64_t           N    = arg.N;
    int64_t           incx = arg.incx;
    int64_t           incy = arg.incy;

    int64_t abs_incx = incx < 0 ? -incx : incx;
    int64_t abs_incy = incy < 0 ? -incy : incy;
    size_t  size_A   = hipblas_packed_matrix_size(N);

    hipblasLocalHandle handle(arg);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy;
    if(invalid_size || !N)
    {
        DAPI_EXPECT(invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS,
                    hipblasSpr2Fn,
                    (handle, uplo, N, nullptr, nullptr, incx, nullptr, incy, nullptr));
        return;
    }

    // Naming: `h` is in CPU (host) memory(eg hAp), `d` is in GPU (device) memory (eg dAp).
    // Allocate host memory
    host_matrix<T> hA(N, N, N);
    host_matrix<T> hAp(1, size_A, 1);
    host_matrix<T> hAp_host(1, size_A, 1);
    host_matrix<T> hAp_device(1, size_A, 1);
    host_matrix<T> hAp_cpu(1, size_A, 1);
    host_vector<T> hx(N, incx);
    host_vector<T> hy(N, incy);

    // Allocate device memory
    device_matrix<T> dAp(1, size_A, 1);
    device_vector<T> dx(N, incx);
    device_vector<T> dy(N, incy);
    device_vector<T> d_alpha(1);

    T h_alpha = arg.get_alpha<T>();

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, hipblas_client_never_set_nan, hipblas_symmetric_matrix, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_vector(hy, arg, hipblas_client_alpha_sets_nan);

    // helper function to convert Regular matrix `hA` to packed matrix `hAp`
    regular_to_packed(uplo == HIPBLAS_FILL_MODE_UPPER, hA, hAp, N);

    hAp_cpu = hAp;

    // copy data from CPU to device
    CHECK_HIP_ERROR(dAp.transfer_from(hAp));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        DAPI_CHECK(hipblasSpr2Fn, (handle, uplo, N, &h_alpha, dx, incx, dy, incy, dAp));

        CHECK_HIP_ERROR(hAp_host.transfer_from(dAp));
        CHECK_HIP_ERROR(dAp.transfer_from(hAp));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        DAPI_CHECK(hipblasSpr2Fn, (handle, uplo, N, d_alpha, dx, incx, dy, incy, dAp));

        CHECK_HIP_ERROR(hAp_device.transfer_from(dAp));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        ref_spr2<T>(uplo, N, h_alpha, hx.data(), incx, hy.data(), incy, hAp_cpu.data());

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(arg.unit_check)
        {
            unit_check_general<T>(1, size_A, 1, hAp_cpu.data(), hAp_host.data());
            unit_check_general<T>(1, size_A, 1, hAp_cpu.data(), hAp_device.data());
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, size_A, 1, hAp_cpu.data(), hAp_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', 1, size_A, 1, hAp_cpu.data(), hAp_device.data());
        }
    }

    if(arg.timing)
    {
        CHECK_HIP_ERROR(dAp.transfer_from(hAp));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            DAPI_DISPATCH(hipblasSpr2Fn, (handle, uplo, N, d_alpha, dx, incx, dy, incy, dAp));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasSpr2Model{}.log_args<T>(std::cout,
                                       arg,
                                       gpu_time_used,
                                       spr2_gflop_count<T>(N),
                                       spr2_gbyte_count<T>(N),
                                       hipblas_error_host,
                                       hipblas_error_device);
    }
}
