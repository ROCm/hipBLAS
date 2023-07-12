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
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

/* ============================================================================================ */

using hipblasGemmBatchedExModel = ArgumentModel<e_transA,
                                                e_transB,
                                                e_M,
                                                e_N,
                                                e_K,
                                                e_alpha,
                                                e_lda,
                                                e_ldb,
                                                e_beta,
                                                e_ldc,
                                                e_batch_count>;

inline void testname_gemm_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
inline hipblasStatus_t testing_gemm_batched_ex_template(const Arguments& arg)
{
    bool FORTRAN                = arg.fortran;
    auto hipblasGemmBatchedExFn = FORTRAN ? hipblasGemmBatchedExFortran : hipblasGemmBatchedEx;

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

    hipblasOperation_t transA = char2hipblas_operation(arg.transA);
    hipblasOperation_t transB = char2hipblas_operation(arg.transB);

    int M = arg.M;
    int N = arg.N;
    int K = arg.K;

    int lda = arg.lda;
    int ldb = arg.ldb;
    int ldc = arg.ldc;

    int batch_count = arg.batch_count;

    hipblasDatatype_t    a_type            = arg.a_type;
    hipblasDatatype_t    b_type            = arg.b_type;
    hipblasDatatype_t    c_type            = arg.c_type;
    hipblasDatatype_t    compute_type      = arg.compute_type;
    hipblasComputeType_t compute_type_gemm = arg.compute_type_gemm;

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
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    const size_t size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    device_batch_vector<Ta> dA(size_A, 1, batch_count);
    device_batch_vector<Tb> dB(size_B, 1, batch_count);
    device_batch_vector<Tc> dC(size_C, 1, batch_count);
    device_vector<Tex>      d_alpha(1);
    device_vector<Tex>      d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    host_batch_vector<Ta> hA(size_A, 1, batch_count);
    host_batch_vector<Tb> hB(size_B, 1, batch_count);
    host_batch_vector<Tc> hC_host(size_C, 1, batch_count);
    host_batch_vector<Tc> hC_device(size_C, 1, batch_count);
    host_batch_vector<Tc> hC_gold(size_C, 1, batch_count);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    hipblas_init_vector(hA, arg, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hB, arg, hipblas_client_alpha_sets_nan);
    hipblas_init_vector(hC_host, arg, hipblas_client_beta_sets_nan);

    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }

    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   &h_alpha_Tex,
                                                   (const void**)(Ta**)dA.ptr_on_device(),
                                                   a_type,
                                                   lda,
                                                   (const void**)(Tb**)dB.ptr_on_device(),
                                                   b_type,
                                                   ldb,
                                                   &h_beta_Tex,
                                                   (void**)(Tc**)dC.ptr_on_device(),
                                                   c_type,
                                                   ldc,
                                                   batch_count,
#ifdef HIPBLAS_V2
                                                   compute_type_gemm,
#else
                                                   compute_type,
#endif
                                                   algo));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));
        CHECK_HIP_ERROR(dC.transfer_from(hC_device));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                   transA,
                                                   transB,
                                                   M,
                                                   N,
                                                   K,
                                                   d_alpha,
                                                   (const void**)(Ta**)dA.ptr_on_device(),
                                                   a_type,
                                                   lda,
                                                   (const void**)(Tb**)dB.ptr_on_device(),
                                                   b_type,
                                                   ldb,
                                                   d_beta,
                                                   (void**)(Tc**)dC.ptr_on_device(),
                                                   c_type,
                                                   ldc,
                                                   batch_count,
#ifdef HIPBLAS_V2
                                                   compute_type_gemm,
#else
                                                   compute_type,
#endif
                                                   algo));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // CPU BLAS
        for(int b = 0; b < batch_count; b++)
        {
            cblas_gemm<Ta, Tc, Tex>(transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    h_alpha_Tex,
                                    hA[b],
                                    lda,
                                    hB[b],
                                    ldb,
                                    h_beta_Tex,
                                    hC_gold[b],
                                    ldc);
        }

        if(unit_check)
        {
            // check for float16/bfloat16 input
            if((getArchMajor() == 11)
               && ((std::is_same<Tex, float>{} && std::is_same<Ta, hipblasBfloat16>{})
                   || (std::is_same<Tex, float>{} && std::is_same<Ta, hipblasHalf>{})
                   || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ta, hipblasHalf>{})))
            {
                const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ta, Tc>;
                near_check_general<Tc>(M, N, batch_count, ldc, hC_gold, hC_host, tol);
                near_check_general<Tc>(M, N, batch_count, ldc, hC_gold, hC_device, tol);
            }
            else
            {
                unit_check_general<Tc>(M, N, batch_count, ldc, hC_gold, hC_host);
                unit_check_general<Tc>(M, N, batch_count, ldc, hC_gold, hC_device);
            }
        }

        if(norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_device, batch_count);
        }
    }

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedExFn(handle,
                                                       transA,
                                                       transB,
                                                       M,
                                                       N,
                                                       K,
                                                       &h_alpha_Tex,
                                                       (const void**)(Ta**)dA.ptr_on_device(),
                                                       a_type,
                                                       lda,
                                                       (const void**)(Tb**)dB.ptr_on_device(),
                                                       b_type,
                                                       ldb,
                                                       &h_beta_Tex,
                                                       (void**)(Tc**)dC.ptr_on_device(),
                                                       c_type,
                                                       ldc,
                                                       batch_count,
#ifdef HIPBLAS_V2
                                                       compute_type_gemm,
#else
                                                       compute_type,
#endif
                                                       algo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmBatchedExModel{}.log_args<Tc>(std::cout,
                                                 arg,
                                                 gpu_time_used,
                                                 gemm_gflop_count<Tex>(M, N, K),
                                                 gemm_gbyte_count<Tex>(M, N, K),
                                                 hipblas_error_host,
                                                 hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

#ifdef HIPBLAS_V2

inline hipblasStatus_t testing_gemm_batched_ex(const Arguments& arg)
{
    // Support is essentially the same with HIPBLAS_V2, but just specified differently with hipblasComputeType_t.
    // Tex is more accurately the precision in which the reference is calculated with.
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    hipblasDatatype_t    a_type            = arg.a_type;
    hipblasDatatype_t    b_type            = arg.b_type;
    hipblasDatatype_t    c_type            = arg.c_type;
    hipblasComputeType_t compute_type_gemm = arg.compute_type_gemm;

    if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_16F
       && compute_type_gemm == HIPBLAS_COMPUTE_16F)
    {
        status = testing_gemm_batched_ex_template<hipblasHalf>(arg);
    }
    else if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_16F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status
            = testing_gemm_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float>(arg);
    }
    else if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasHalf, hipblasHalf, float, float>(arg);
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF && c_type == HIP_R_16BF
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasBfloat16,
                                                  hipblasBfloat16,
                                                  hipblasBfloat16,
                                                  float>(arg);
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status
            = testing_gemm_batched_ex_template<hipblasBfloat16, hipblasBfloat16, float, float>(arg);
    }
    else if(a_type == HIP_R_32F && b_type == HIP_R_32F && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_batched_ex_template<float>(arg);
    }
    else if(a_type == HIP_R_64F && b_type == HIP_R_64F && c_type == HIP_R_64F
            && compute_type_gemm == HIPBLAS_COMPUTE_64F)
    {
        status = testing_gemm_batched_ex_template<double>(arg);
    }
    else if(a_type == HIP_R_8I && b_type == HIP_R_8I && c_type == HIP_R_32I
            && compute_type_gemm == HIPBLAS_COMPUTE_32I)
    {
        status = testing_gemm_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(arg);
    }
    else if(a_type == HIP_C_32F && b_type == HIP_C_32F && c_type == HIP_C_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasComplex>(arg);
    }
    else if(a_type == HIP_C_64F && b_type == HIP_C_64F && c_type == HIP_C_64F
            && compute_type_gemm == HIPBLAS_COMPUTE_64F)
    {
        status = testing_gemm_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

#else

inline hipblasStatus_t testing_gemm_batched_ex(const Arguments& arg)
{
    hipblasDatatype_t a_type       = arg.a_type;
    hipblasDatatype_t b_type       = arg.b_type;
    hipblasDatatype_t c_type       = arg.c_type;
    hipblasDatatype_t compute_type = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_batched_ex_template<hipblasHalf>(arg);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && compute_type == HIPBLAS_R_32F)
    {
        status
            = testing_gemm_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float>(arg);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasHalf, hipblasHalf, float, float>(arg);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_16B
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasBfloat16,
                                                  hipblasBfloat16,
                                                  hipblasBfloat16,
                                                  float>(arg);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status
            = testing_gemm_batched_ex_template<hipblasBfloat16, hipblasBfloat16, float, float>(arg);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_batched_ex_template<float>(arg);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_batched_ex_template<double>(arg);
    }
    else if(a_type == HIPBLAS_C_32F && b_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F
            && compute_type == HIPBLAS_C_32F)
    {
        status = testing_gemm_batched_ex_template<hipblasComplex>(arg);
    }
    else if(a_type == HIPBLAS_C_64F && b_type == HIPBLAS_C_64F && c_type == HIPBLAS_C_64F
            && compute_type == HIPBLAS_C_64F)
    {
        status = testing_gemm_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else if(a_type == HIPBLAS_R_8I && b_type == HIPBLAS_R_8I && c_type == HIPBLAS_R_32I
            && compute_type == HIPBLAS_R_32I)
    {
        status = testing_gemm_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

#endif
