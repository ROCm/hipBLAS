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

using hipblasGemmStridedBatchedExModel = ArgumentModel<e_transA,
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
// strides not logged

inline void testname_gemm_strided_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmStridedBatchedExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
inline hipblasStatus_t testing_gemm_strided_batched_ex_template(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasGemmStridedBatchedExFn
        = FORTRAN ? hipblasGemmStridedBatchedExFortran : hipblasGemmStridedBatchedExFortran;
#ifdef HIPBLAS_V2
    auto hipblasGemmStridedBatchedExWithFlagsFn = FORTRAN
                                                      ? hipblasGemmStridedBatchedExWithFlagsFortran
                                                      : hipblasGemmStridedBatchedExWithFlags;
#endif

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

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

    int batch_count = arg.batch_count;

    int norm_check = arg.norm_check;
    int unit_check = arg.unit_check;
    int timing     = arg.timing;

    Tex h_alpha_Tex = arg.get_alpha<Tex>();
    Tex h_beta_Tex  = arg.get_beta<Tex>();

    int A_row = transA == HIPBLAS_OP_N ? M : K;
    int A_col = transA == HIPBLAS_OP_N ? K : M;
    int B_row = transB == HIPBLAS_OP_N ? K : N;
    int B_col = transB == HIPBLAS_OP_N ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    const size_t stride_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t stride_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t stride_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    const size_t size_A = stride_A * batch_count;
    const size_t size_B = stride_B * batch_count;
    const size_t size_C = stride_C * batch_count;

    device_vector<Ta> dA(size_A);
    device_vector<Tb> dB(size_B);
    device_vector<Tc> dC(size_C);

    device_vector<Tex> d_alpha(1);
    device_vector<Tex> d_beta(1);

    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ta> hA(size_A);
    host_vector<Tb> hB(size_B);
    host_vector<Tc> hC_host(size_C);
    host_vector<Tc> hC_device(size_C);
    host_vector<Tc> hC_gold(size_C);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, arg, A_row, A_col, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, arg, B_row, B_col, ldb, stride_B, batch_count, hipblas_client_alpha_sets_nan);
    hipblas_init_matrix(
        hC_host, arg, M, N, ldc, stride_C, batch_count, hipblas_client_beta_sets_nan);
    hC_gold = hC_device = hC_host;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Tb) * size_B, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(Tc) * size_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tex, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tex, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        if(!arg.with_flags)
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              &h_alpha_Tex,
                                                              dA,
                                                              a_type,
                                                              lda,
                                                              stride_A,
                                                              dB,
                                                              b_type,
                                                              ldb,
                                                              stride_B,
                                                              &h_beta_Tex,
                                                              dC,
                                                              c_type,
                                                              ldc,
                                                              stride_C,
                                                              batch_count,
#ifdef HIPBLAS_V2
                                                              compute_type_gemm,
#else
                                                              compute_type,
#endif
                                                              algo));
        }
#ifdef HIPBLAS_V2
        else
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExWithFlagsFn(handle,
                                                                       transA,
                                                                       transB,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       &h_alpha_Tex,
                                                                       dA,
                                                                       a_type,
                                                                       lda,
                                                                       stride_A,
                                                                       dB,
                                                                       b_type,
                                                                       ldb,
                                                                       stride_B,
                                                                       &h_beta_Tex,
                                                                       dC,
                                                                       c_type,
                                                                       ldc,
                                                                       stride_C,
                                                                       batch_count,
                                                                       compute_type_gemm,
                                                                       algo,
                                                                       flags));
        }
#endif

        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(Tc) * size_C, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        if(!arg.with_flags)
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              d_alpha,
                                                              dA,
                                                              a_type,
                                                              lda,
                                                              stride_A,
                                                              dB,
                                                              b_type,
                                                              ldb,
                                                              stride_B,
                                                              d_beta,
                                                              dC,
                                                              c_type,
                                                              ldc,
                                                              stride_C,
                                                              batch_count,
#ifdef HIPBLAS_V2
                                                              compute_type_gemm,
#else
                                                              compute_type,
#endif
                                                              algo));
        }
#ifdef HIPBLAS_V2
        else
        {
            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExWithFlagsFn(handle,
                                                                       transA,
                                                                       transB,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       d_alpha,
                                                                       dA,
                                                                       a_type,
                                                                       lda,
                                                                       stride_A,
                                                                       dB,
                                                                       b_type,
                                                                       ldb,
                                                                       stride_B,
                                                                       d_beta,
                                                                       dC,
                                                                       c_type,
                                                                       ldc,
                                                                       stride_C,
                                                                       batch_count,
                                                                       compute_type_gemm,
                                                                       algo,
                                                                       flags));
        }
#endif

        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        for(int b = 0; b < batch_count; b++)
        {
            cblas_gemm<Ta, Tc, Tex>(transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    h_alpha_Tex,
                                    hA.data() + b * stride_A,
                                    lda,
                                    hB.data() + b * stride_B,
                                    ldb,
                                    h_beta_Tex,
                                    hC_gold.data() + b * stride_C,
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
                near_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_host, tol);
                near_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_device, tol);
            }
            else
            {
                unit_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_host);
                unit_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_device);
            }
        }
        if(arg.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tc>('F', M, N, ldc, stride_C, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<Tc>('F', M, N, ldc, stride_C, hC_gold, hC_device, batch_count);
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

            if(!arg.with_flags)
            {
                CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExFn(handle,
                                                                  transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  &h_alpha_Tex,
                                                                  dA,
                                                                  a_type,
                                                                  lda,
                                                                  stride_A,
                                                                  dB,
                                                                  b_type,
                                                                  ldb,
                                                                  stride_B,
                                                                  &h_beta_Tex,
                                                                  dC,
                                                                  c_type,
                                                                  ldc,
                                                                  stride_C,
                                                                  batch_count,
#ifdef HIPBLAS_V2
                                                                  compute_type_gemm,
#else
                                                                  compute_type,
#endif
                                                                  algo));
            }
#ifdef HIPBLAS_V2
            else
            {
                CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExWithFlagsFn(handle,
                                                                           transA,
                                                                           transB,
                                                                           M,
                                                                           N,
                                                                           K,
                                                                           &h_alpha_Tex,
                                                                           dA,
                                                                           a_type,
                                                                           lda,
                                                                           stride_A,
                                                                           dB,
                                                                           b_type,
                                                                           ldb,
                                                                           stride_B,
                                                                           &h_beta_Tex,
                                                                           dC,
                                                                           c_type,
                                                                           ldc,
                                                                           stride_C,
                                                                           batch_count,
                                                                           compute_type_gemm,
                                                                           algo,
                                                                           flags));
            }
#endif
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmStridedBatchedExModel{}.log_args<Tc>(std::cout,
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

inline hipblasStatus_t testing_gemm_strided_batched_ex(const Arguments& arg)
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
        status = testing_gemm_strided_batched_ex_template<hipblasHalf>(arg);
    }
    else if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_16F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf,
                                                          hipblasHalf,
                                                          hipblasHalf,
                                                          float>(arg);
    }
    else if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status
            = testing_gemm_strided_batched_ex_template<hipblasHalf, hipblasHalf, float, float>(arg);
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF && c_type == HIP_R_16BF
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          float>(arg);
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          float,
                                                          float>(arg);
    }
    else if(a_type == HIP_R_32F && b_type == HIP_R_32F && c_type == HIP_R_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_strided_batched_ex_template<float>(arg);
    }
    else if(a_type == HIP_R_64F && b_type == HIP_R_64F && c_type == HIP_R_64F
            && compute_type_gemm == HIPBLAS_COMPUTE_64F)
    {
        status = testing_gemm_strided_batched_ex_template<double>(arg);
    }
    else if(a_type == HIP_R_8I && b_type == HIP_R_8I && c_type == HIP_R_32I
            && compute_type_gemm == HIPBLAS_COMPUTE_32I)
    {
        status = testing_gemm_strided_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(arg);
    }
    else if(a_type == HIP_C_32F && b_type == HIP_C_32F && c_type == HIP_C_32F
            && compute_type_gemm == HIPBLAS_COMPUTE_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasComplex>(arg);
    }
    else if(a_type == HIP_C_64F && b_type == HIP_C_64F && c_type == HIP_C_64F
            && compute_type_gemm == HIPBLAS_COMPUTE_64F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

#else

inline hipblasStatus_t testing_gemm_strided_batched_ex(const Arguments& arg)
{
    hipblasDatatype_t a_type = arg.a_type;
    hipblasDatatype_t b_type = arg.b_type;
    hipblasDatatype_t c_type = arg.c_type;
    hipblasDatatype_t compute_type = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf>(arg);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf,
                                                          hipblasHalf,
                                                          hipblasHalf,
                                                          float>(arg);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status
            = testing_gemm_strided_batched_ex_template<hipblasHalf, hipblasHalf, float, float>(arg);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_16B
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          float>(arg);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          float,
                                                          float>(arg);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<float>(arg);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_strided_batched_ex_template<double>(arg);
    }
    else if(a_type == HIPBLAS_C_32F && b_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F
            && compute_type == HIPBLAS_C_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasComplex>(arg);
    }
    else if(a_type == HIPBLAS_C_64F && b_type == HIPBLAS_C_64F && c_type == HIPBLAS_C_64F
            && compute_type == HIPBLAS_C_64F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else if(a_type == HIPBLAS_R_8I && b_type == HIPBLAS_R_8I && c_type == HIPBLAS_R_32I
            && compute_type == HIPBLAS_R_32I)
    {
        status = testing_gemm_strided_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

#endif
