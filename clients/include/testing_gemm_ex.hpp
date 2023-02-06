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

using hipblasGemmExModel
    = ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_alpha, e_lda, e_ldb, e_beta, e_ldc>;

inline void testname_gemm_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmExModel{}.test_name(arg, name);
}

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
inline hipblasStatus_t testing_gemm_ex_template(const Arguments& arg)
{
    bool FORTRAN         = arg.fortran;
    auto hipblasGemmExFn = FORTRAN ? hipblasGemmExFortran : hipblasGemmEx;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    uint32_t          solution_index = 0;
    uint32_t          flags          = 0;
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

    hipDataType a_type       = arg.a_type;
    hipDataType b_type       = arg.b_type;
    hipDataType c_type       = arg.c_type;
    hipDataType compute_type = arg.compute_type;

    Tex h_alpha_Tc = arg.get_alpha<Tex>();
    Tex h_beta_Tc  = arg.get_beta<Tex>();

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
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    const size_t size_A = static_cast<size_t>(lda) * static_cast<size_t>(A_col);
    const size_t size_B = static_cast<size_t>(ldb) * static_cast<size_t>(B_col);
    const size_t size_C = static_cast<size_t>(ldc) * static_cast<size_t>(N);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ta> hA(size_A);
    host_vector<Tb> hB(size_B);
    host_vector<Tc> hC_host(size_C);
    host_vector<Tc> hC_device(size_C);
    host_vector<Tc> hC_gold(size_C);

    device_vector<Ta>  dA(size_A);
    device_vector<Tb>  dB(size_B);
    device_vector<Tc>  dC(size_C);
    device_vector<Tex> d_alpha(1);
    device_vector<Tex> d_beta(1);

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    for(auto int8Type : {HIPBLAS_INT8_DATATYPE_DEFAULT,
                         HIPBLAS_INT8_DATATYPE_INT8,
                         HIPBLAS_INT8_DATATYPE_PACK_INT8x4})
    {
        // only need to test multiple int8Type for int8_t, for other datatypes break
        if(!(std::is_same<Ta, int8_t>{}) && HIPBLAS_INT8_DATATYPE_DEFAULT != int8Type)
            break;

        hipblasSetInt8Datatype(handle, int8Type);

        // Initial Data on CPU
        hipblas_init_matrix(hA, arg, A_row, A_col, lda, 0, 1, hipblas_client_alpha_sets_nan, true);
        hipblas_init_matrix(
            hB, arg, B_row, B_col, ldb, 0, 1, hipblas_client_alpha_sets_nan, false, true);
        hipblas_init_matrix(hC_host, arg, M, N, ldc, 0, 1, hipblas_client_beta_sets_nan);

        hC_gold = hC_device = hC_host;

        // copy data from CPU to device

        // CUDA doesn't do packing
#ifdef __HIP_PLATFORM_NVCC__
        if(HIPBLAS_INT8_DATATYPE_DEFAULT != int8Type)
            break;
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Tb) * size_B, hipMemcpyHostToDevice));
#else
        if(std::is_same<Ta, int8_t>{} && transA == HIPBLAS_OP_N && layout_pack_int8(handle))
        {
            host_vector<Ta> hA_packed(hA);
            hipblas_packInt8(hA_packed, M, K, lda);
            CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
        }
        else
        {
            CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
        }

        if(std::is_same<Tb, int8_t>{} && transB != HIPBLAS_OP_N && layout_pack_int8(handle))
        {
            host_vector<Tb> hB_packed(hB);
            hipblas_packInt8(hB_packed, N, K, ldb);
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
            CHECK_HIPBLAS_ERROR(hipblasGemmExFn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &h_alpha_Tc,
                                                dA,
                                                a_type,
                                                lda,
                                                dB,
                                                b_type,
                                                ldb,
                                                &h_beta_Tc,
                                                dC,
                                                c_type,
                                                ldc,
                                                compute_type,
                                                algo));

            CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(Tc) * size_C, hipMemcpyHostToDevice));

            CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
            CHECK_HIPBLAS_ERROR(hipblasGemmExFn(handle,
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
                                                compute_type,
                                                algo));

            CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));

            // reference BLAS
            cblas_gemm<Ta, Tc, Tex>(transA,
                                    transB,
                                    M,
                                    N,
                                    K,
                                    h_alpha_Tc,
                                    hA.data(),
                                    lda,
                                    hB.data(),
                                    ldb,
                                    h_beta_Tc,
                                    hC_gold.data(),
                                    ldc);

            if(unit_check)
            {
                // check for float16/bfloat16 input
                if((getArchMajor() == 11)
                   && ((std::is_same<Tex, float>{} && std::is_same<Ta, hipblasBfloat16>{})
                       || (std::is_same<Tex, float>{} && std::is_same<Ta, hipblasHalf>{})
                       || (std::is_same<Tex, hipblasHalf>{} && std::is_same<Ta, hipblasHalf>{})))
                {
                    const double tol = K * sum_error_tolerance_for_gfx11<Tex, Ta, Tc>;
                    near_check_general<Tc>(M, N, ldc, hC_gold.data(), hC_host.data(), tol);
                    near_check_general<Tc>(M, N, ldc, hC_gold.data(), hC_device.data(), tol);
                }
                else
                {
                    unit_check_general<Tc>(M, N, ldc, hC_gold, hC_host);
                    unit_check_general<Tc>(M, N, ldc, hC_gold, hC_device);
                }
            }
            if(norm_check)
            {
                hipblas_error_host
                    = std::abs(norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_host));
                hipblas_error_device
                    = std::abs(norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_device));
            }
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

            CHECK_HIPBLAS_ERROR(hipblasGemmExFn(handle,
                                                transA,
                                                transB,
                                                M,
                                                N,
                                                K,
                                                &h_alpha_Tc,
                                                dA,
                                                a_type,
                                                lda,
                                                dB,
                                                b_type,
                                                ldb,
                                                &h_beta_Tc,
                                                dC,
                                                c_type,
                                                ldc,
                                                compute_type,
                                                algo));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        hipblasGemmExModel{}.log_args<Tc>(std::cout,
                                          arg,
                                          gpu_time_used,
                                          gemm_gflop_count<Tex>(M, N, K),
                                          gemm_gbyte_count<Tex>(M, N, K),
                                          hipblas_error_host,
                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

inline hipblasStatus_t testing_gemm_ex(const Arguments& arg)
{
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    hipDataType a_type       = arg.a_type;
    hipDataType b_type       = arg.b_type;
    hipDataType c_type       = arg.c_type;
    hipDataType compute_type = arg.compute_type;

    if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_16F && c_type == HIP_R_16F
       && compute_type == HIP_R_16F)
    {
        status = testing_gemm_ex_template<hipblasHalf>(arg);
    }
    else if(a_type == HIP_R_16F && b_type == HIP_R_16F && c_type == HIP_R_16F && c_type == HIP_R_16F
            && compute_type == HIP_R_32F)
    {
        status = testing_gemm_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float>(arg);
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF && c_type == HIP_R_16BF
            && c_type == HIP_R_16BF && compute_type == HIP_R_32F)
    {
        status = testing_gemm_ex_template<hipblasBfloat16, hipblasBfloat16, hipblasBfloat16, float>(
            arg);
    }
    else if(a_type == HIP_R_32F && b_type == HIP_R_32F && c_type == HIP_R_32F && c_type == HIP_R_32F
            && compute_type == HIP_R_32F)
    {
        status = testing_gemm_ex_template<float>(arg);
    }
    else if(a_type == HIP_R_64F && b_type == HIP_R_64F && c_type == HIP_R_64F && c_type == HIP_R_64F
            && compute_type == HIP_R_64F)
    {
        status = testing_gemm_ex_template<double>(arg);
    }
    else if(a_type == HIP_C_32F && b_type == HIP_C_32F && c_type == HIP_C_32F && c_type == HIP_C_32F
            && compute_type == HIP_C_32F)
    {
        status = testing_gemm_ex_template<hipblasComplex>(arg);
    }
    else if(a_type == HIP_C_64F && b_type == HIP_C_64F && c_type == HIP_C_64F && c_type == HIP_C_64F
            && compute_type == HIP_C_64F)
    {
        status = testing_gemm_ex_template<hipblasDoubleComplex>(arg);
    }
    else if(a_type == HIP_R_8I && b_type == HIP_R_8I && c_type == HIP_R_32I && c_type == HIP_R_32I
            && compute_type == HIP_R_32I)
    {
        status = testing_gemm_ex_template<int8_t, int8_t, int32_t, int32_t>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
