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

#include "utility.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGemmExModel = ArgumentModel<e_a_type,
                                         e_c_type,
                                         e_compute_type,
                                         e_transA,
                                         e_transB,
                                         e_M,
                                         e_N,
                                         e_K,
                                         e_alpha,
                                         e_lda,
                                         e_ldb,
                                         e_beta,
                                         e_ldc,
                                         e_with_flags,
                                         e_flags>;

inline void testname_gemm_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmExModel{}.test_name(arg, name);
}

template <typename Ti, typename To = Ti, typename Tex = To>
void testing_gemm_ex(const Arguments& arg)
{
    bool FORTRAN         = arg.fortran;
    auto hipblasGemmExFn = FORTRAN ? hipblasGemmExFortran : hipblasGemmEx;
    auto hipblasGemmExWithFlagsFn
        = FORTRAN ? hipblasGemmExWithFlagsFortran : hipblasGemmExWithFlags;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    size_t*           workspace_size = 0;
    void*             workspace      = 0;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB = char2hipblas_operation(arg.transB);
    int                M      = arg.M;
    int                N      = arg.N;
    int                K      = arg.K;
    int                lda    = arg.lda;
    int                ldb    = arg.ldb;
    int                ldc    = arg.ldc;

    hipblasDatatype_t    a_type            = arg.a_type;
    hipblasDatatype_t    b_type            = arg.b_type;
    hipblasDatatype_t    c_type            = arg.c_type;
    hipblasDatatype_t    compute_type      = arg.compute_type;
    hipblasComputeType_t compute_type_gemm = arg.compute_type_gemm;
    hipblasGemmFlags_t   flags             = hipblasGemmFlags_t(arg.flags);

    Tex h_alpha_Tex = arg.get_alpha<Tex>();
    Tex h_beta_Tex  = arg.get_beta<Tex>();

    int norm_check = arg.norm_check;
    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    int A_row = transA == HIPBLAS_OP_N ? M : K;
    int A_col = transA == HIPBLAS_OP_N ? K : M;
    int B_row = transB == HIPBLAS_OP_N ? K : N;
    int B_col = transB == HIPBLAS_OP_N ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        return;
    }

    const size_t size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti> hA(size_A);
    host_vector<Ti> hB(size_B);
    host_vector<To> hC_host(size_C);
    host_vector<To> hC_device(size_C);
    host_vector<To> hC_gold(size_C);

    device_vector<Ti>  dA(size_A);
    device_vector<Ti>  dB(size_B);
    device_vector<To>  dC(size_C);
    device_vector<Tex> d_alpha(1);
    device_vector<Tex> d_beta(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Initial Data on CPU
    hipblas_init_matrix(hA, arg, A_row, A_col, lda, 0, 1, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, B_row, B_col, ldb, 0, 1, hipblas_client_alpha_sets_nan, false, true);
    hipblas_init_matrix(hC_host, arg, M, N, ldc, 0, 1, hipblas_client_beta_sets_nan);

    hC_gold = hC_device = hC_host;

    // copy data from CPU to device

    ASSERT_HIP_SUCCESS(hipMemcpy(dA, hA, sizeof(Ti) * size_A, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(dB, hB, sizeof(Ti) * size_B, hipMemcpyHostToDevice));

    ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_host, sizeof(To) * size_C, hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        if(!arg.with_flags)
        {
            ASSERT_HIPBLAS_SUCCESS(hipblasGemmExFn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   &h_alpha_Tex,
                                                   dA,
                                                   a_type,
                                                   lda,
                                                   dB,
                                                   b_type,
                                                   ldb,
                                                   &h_beta_Tex,
                                                   dC,
                                                   c_type,
                                                   ldc,
#ifdef HIPBLAS_V2
                                                   compute_type_gemm,
#else
                                                   compute_type,
#endif
                                                   algo));
        }
        else
        {
            ASSERT_HIPBLAS_SUCCESS(hipblasGemmExWithFlagsFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha_Tex,
                                                            dA,
                                                            a_type,
                                                            lda,
                                                            dB,
                                                            b_type,
                                                            ldb,
                                                            &h_beta_Tex,
                                                            dC,
                                                            c_type,
                                                            ldc,
#ifdef HIPBLAS_V2
                                                            compute_type_gemm,
#else
                                                            compute_type,
#endif
                                                            algo,
                                                            flags));
        }

        ASSERT_HIP_SUCCESS(hipMemcpy(hC_host, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));
        ASSERT_HIP_SUCCESS(hipMemcpy(dC, hC_device, sizeof(To) * size_C, hipMemcpyHostToDevice));

        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        if(!arg.with_flags)
        {
            ASSERT_HIPBLAS_SUCCESS(hipblasGemmExFn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   d_alpha,
                                                   dA,
                                                   a_type,
                                                   lda,
                                                   dB,
                                                   b_type,
                                                   ldb,
                                                   d_beta,
                                                   dC,
                                                   c_type,
                                                   ldc,
#ifdef HIPBLAS_V2
                                                   compute_type_gemm,
#else
                                                   compute_type,
#endif
                                                   algo));
        }
        else
        {
            ASSERT_HIPBLAS_SUCCESS(hipblasGemmExWithFlagsFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            a_type,
                                                            lda,
                                                            dB,
                                                            b_type,
                                                            ldb,
                                                            d_beta,
                                                            dC,
                                                            c_type,
                                                            ldc,
#ifdef HIPBLAS_V2
                                                            compute_type_gemm,
#else
                                                            compute_type,
#endif
                                                            algo,
                                                            flags));
        }

        ASSERT_HIP_SUCCESS(hipMemcpy(hC_device, dC, sizeof(To) * size_C, hipMemcpyDeviceToHost));

        // reference BLAS
        cblas_gemm<Ti, To, Tex>(transA,
                                transB,
                                M,
                                N,
                                K,
                                h_alpha_Tex,
                                hA.data(),
                                lda,
                                hB.data(),
                                ldb,
                                h_beta_Tex,
                                hC_gold.data(),
                                ldc);

        if(unit_check)
        {
            // check for float16/bfloat16 input
            if((getArchMajor() == 11)
               && ((std::is_same<Tex, float>{} && std::is_same<Ti, hipblasBfloat16>{})
                   || (std::is_same<Tex, float>{} && std::is_same<Ti, hipblasHalf>{})
                   || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ti, hipblasHalf>{})))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ti, To>;
                near_check_general<To>(M, N, ldc, hC_gold.data(), hC_host.data(), tol);
                near_check_general<To>(M, N, ldc, hC_gold.data(), hC_device.data(), tol);
            }
            else
            {
                unit_check_general<To>(M, N, ldc, hC_gold, hC_host);
                unit_check_general<To>(M, N, ldc, hC_gold, hC_device);
            }
        }
        if(norm_check)
        {
            hipblas_error_host
                = hipblas_abs(norm_check_general<To>('F', M, N, ldc, hC_gold, hC_host));
            hipblas_error_device
                = hipblas_abs(norm_check_general<To>('F', M, N, ldc, hC_gold, hC_device));
        }
    }

    if(timing)
    {
        hipStream_t stream;
        ASSERT_HIPBLAS_SUCCESS(hipblasGetStream(handle, &stream));
        ASSERT_HIPBLAS_SUCCESS(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            if(!arg.with_flags)
            {
                ASSERT_HIPBLAS_SUCCESS(hipblasGemmExFn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tex,
                                                       dA,
                                                       a_type,
                                                       lda,
                                                       dB,
                                                       b_type,
                                                       ldb,
                                                       &h_beta_Tex,
                                                       dC,
                                                       c_type,
                                                       ldc,
#ifdef HIPBLAS_V2
                                                       compute_type_gemm,
#else
                                                       compute_type,
#endif
                                                       algo));
            }
            else
            {
                ASSERT_HIPBLAS_SUCCESS(hipblasGemmExWithFlagsFn(handle,
                                                                transA,
                                                                transB,
                                                                M,
                                                                N,
                                                                K,
                                                                &h_alpha_Tex,
                                                                dA,
                                                                a_type,
                                                                lda,
                                                                dB,
                                                                b_type,
                                                                ldb,
                                                                &h_beta_Tex,
                                                                dC,
                                                                c_type,
                                                                ldc,
#ifdef HIPBLAS_V2
                                                                compute_type_gemm,
#else
                                                                compute_type,
#endif
                                                                algo,
                                                                flags));
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmExModel{}.log_args<To>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          gemm_gflop_count<Tex>(M, N, K),
                                          gemm_gbyte_count<Tex>(M, N, K),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }
}

template <typename Ti, typename To = Ti, typename Tex = To>
hipblasStatus_t testing_gemm_ex_ret(const Arguments& arg)
{
    testing_gemm_ex<Ti, To, Tex>(arg);
    return HIPBLAS_STATUS_SUCCESS;
}
