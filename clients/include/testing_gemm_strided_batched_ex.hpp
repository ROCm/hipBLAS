/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
hipblasStatus_t testing_gemm_strided_batched_ex_template(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemmStridedBatchedExFn
        = FORTRAN ? hipblasGemmStridedBatchedExFortran : hipblasGemmStridedBatchedExFortran;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    uint32_t          solution_index = 0;
    uint32_t          flags          = 0;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    hipblasDatatype_t a_type       = argus.a_type;
    hipblasDatatype_t b_type       = argus.b_type;
    hipblasDatatype_t c_type       = argus.c_type;
    hipblasDatatype_t compute_type = argus.compute_type;

    int batch_count = argus.batch_count;

    int norm_check = argus.norm_check;
    int unit_check = argus.unit_check;
    int timing     = argus.timing;

    Tex h_alpha_Tc = argus.get_alpha<Tex>();
    Tex h_beta_Tc  = argus.get_beta<Tex>();

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
    hipblasLocalHandle handle(argus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ta> hA(size_A);
    host_vector<Tb> hB(size_B);
    host_vector<Tc> hC_host(size_C);
    host_vector<Tc> hC_device(size_C);
    host_vector<Tc> hC_gold(size_C);

    // Initial Data on CPU
    hipblas_init_matrix(
        hA, argus, A_row, A_col, lda, stride_A, batch_count, hipblas_client_alpha_sets_nan, true);
    hipblas_init_matrix(
        hB, argus, B_row, B_col, ldb, stride_B, batch_count, hipblas_client_alpha_sets_nan);
    hipblas_init_matrix(
        hC_host, argus, M, N, ldc, stride_C, batch_count, hipblas_client_beta_sets_nan);
    hC_gold = hC_device = hC_host;

    // copy data from CPU to device
#ifdef __HIP_PLATFORM_NVCC__
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Tb) * size_B, hipMemcpyHostToDevice));
#else
    if(std::is_same<Ta, int8_t>{} && transA == HIPBLAS_OP_N && layout_pack_int8(handle))
    {
        host_vector<Ta> hA_packed(hA);
        hipblas_packInt8(hA_packed, M, K, lda, batch_count, stride_A);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }

    if(std::is_same<Tb, int8_t>{} && transB != HIPBLAS_OP_N && layout_pack_int8(handle))
    {
        host_vector<Tb> hB_packed(hB);
        hipblas_packInt8(hB_packed, N, K, ldb, batch_count, stride_B);
        CHECK_HIP_ERROR(hipMemcpy(dB, hB_packed, sizeof(Tb) * size_B, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Tb) * size_B, hipMemcpyHostToDevice));
    }
#endif

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(Tc) * size_C, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha_Tc, sizeof(Tex), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta_Tc, sizeof(Tex), hipMemcpyHostToDevice));

    if(unit_check || norm_check)
    {
        // hipBLAS
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExFn(handle,
                                                          transA,
                                                          transB,
                                                          M,
                                                          N,
                                                          K,
                                                          &h_alpha_Tc,
                                                          dA,
                                                          a_type,
                                                          lda,
                                                          stride_A,
                                                          dB,
                                                          b_type,
                                                          ldb,
                                                          stride_B,
                                                          &h_beta_Tc,
                                                          dC,
                                                          c_type,
                                                          ldc,
                                                          stride_C,
                                                          batch_count,
                                                          compute_type,
                                                          algo));

        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(Tc) * size_C, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
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
                                                          compute_type,
                                                          algo));

        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));

        // CPU BLAS
        for(int b = 0; b < batch_count; b++)
        {
            cblas_gemm<Ta, Tc, Tex>(transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    h_alpha_Tc,
                                    hA.data() + b * stride_A,
                                    lda,
                                    hB.data() + b * stride_B,
                                    ldb,
                                    h_beta_Tc,
                                    hC_gold.data() + b * stride_C,
                                    ldc);
        }

        if(argus.unit_check)
        {
            unit_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_host);
            unit_check_general<Tc>(M, N, batch_count, ldc, stride_C, hC_gold, hC_device);
        }
        if(argus.norm_check)
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

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedExFn(handle,
                                                              transA,
                                                              transB,
                                                              M,
                                                              N,
                                                              K,
                                                              &h_alpha_Tc,
                                                              dA,
                                                              a_type,
                                                              lda,
                                                              stride_A,
                                                              dB,
                                                              b_type,
                                                              ldb,
                                                              stride_B,
                                                              &h_beta_Tc,
                                                              dC,
                                                              c_type,
                                                              ldc,
                                                              stride_C,
                                                              batch_count,
                                                              compute_type,
                                                              algo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA_option,
                      e_transB_option,
                      e_M,
                      e_N,
                      e_K,
                      e_lda,
                      e_ldb,
                      e_ldc,
                      e_batch_count>{}
            .log_args<Tc>(std::cout,
                          argus,
                          gpu_time_used,
                          gemm_gflop_count<Tex>(M, N, K),
                          gemm_gbyte_count<Tex>(M, N, K),
                          hipblas_error_host,
                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_gemm_strided_batched_ex(const Arguments& argus)
{
    hipblasDatatype_t a_type       = argus.a_type;
    hipblasDatatype_t b_type       = argus.b_type;
    hipblasDatatype_t c_type       = argus.c_type;
    hipblasDatatype_t compute_type = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf>(argus);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf,
                                                          hipblasHalf,
                                                          hipblasHalf,
                                                          float>(argus);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_16B
            && c_type == HIPBLAS_R_16B && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          hipblasBfloat16,
                                                          float>(argus);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && c_type == HIPBLAS_R_32F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<float>(argus);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && c_type == HIPBLAS_R_64F && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_strided_batched_ex_template<double>(argus);
    }
    else if(a_type == HIPBLAS_C_32F && b_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F
            && c_type == HIPBLAS_C_32F && compute_type == HIPBLAS_C_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasComplex>(argus);
    }
    else if(a_type == HIPBLAS_C_64F && b_type == HIPBLAS_C_64F && c_type == HIPBLAS_C_64F
            && c_type == HIPBLAS_C_64F && compute_type == HIPBLAS_C_64F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else if(a_type == HIPBLAS_R_8I && b_type == HIPBLAS_R_8I && c_type == HIPBLAS_R_32I
            && c_type == HIPBLAS_R_32I && compute_type == HIPBLAS_R_32I)
    {
        status = testing_gemm_strided_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
