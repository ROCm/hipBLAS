/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <sys/time.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
hipblasStatus_t testing_gemm_strided_batched_ex_template(hipblasOperation_t transA,
                                                         hipblasOperation_t transB,
                                                         int                M,
                                                         int                N,
                                                         int                K,
                                                         float              alpha_float,
                                                         int                lda,
                                                         int                ldb,
                                                         float              beta_float,
                                                         int                ldc,
                                                         int                norm_check,
                                                         int                unit_check,
                                                         hipblasDatatype_t  a_type,
                                                         hipblasDatatype_t  b_type,
                                                         hipblasDatatype_t  c_type,
                                                         int                batch_count,
                                                         hipblasDatatype_t  compute_type,
                                                         bool               FORTRAN)
{
    auto hipblasGemmStridedBatchedExFn
        = FORTRAN ? hipblasGemmStridedBatchedExFortran : hipblasGemmStridedBatchedExFortran;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    uint32_t          solution_index = 0;
    uint32_t          flags          = 0;

    Tex h_alpha_Tc;
    Tex h_beta_Tc;

    if(is_same<Tex, hipblasHalf>::value)
    {
        h_alpha_Tc = float_to_half(alpha_float);
        h_beta_Tc  = float_to_half(beta_float);
    }
    else if(is_same<Tex, float>::value)
    {
        h_alpha_Tc = static_cast<Tex>(alpha_float);
        h_beta_Tc  = static_cast<Tex>(beta_float);
    }
    else if(is_same<Tex, double>::value)
    {
        h_alpha_Tc = static_cast<Tex>(alpha_float);
        h_beta_Tc  = static_cast<Tex>(beta_float);
    }
    else if(is_same<Tex, hipblasComplex>::value)
    {
        h_alpha_Tc = static_cast<Tex>(alpha_float);
        h_beta_Tc  = static_cast<Tex>(beta_float);
    }
    else if(is_same<Tex, hipblasDoubleComplex>::value)
    {
        h_alpha_Tc = static_cast<Tex>(alpha_float);
        h_beta_Tc  = static_cast<Tex>(beta_float);
    }
    else if(is_same<Tex, int32_t>::value)
    {
        h_alpha_Tc = static_cast<Tex>(alpha_float);
        h_beta_Tc  = static_cast<Tex>(beta_float);
    }
    else
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

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

    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);

    if(!dA || !dB || !dC || !d_alpha_Tc || !d_beta_Tc)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    hipblasHandle_t handle;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ta> hA(size_A);
    host_vector<Tb> hB(size_B);
    host_vector<Tc> hC(size_C);
    host_vector<Tc> hC_gold(size_C);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Ta>(hA, A_row, A_col, lda, stride_A, batch_count);
    hipblas_init_alternating_sign<Tb>(hB, B_row, B_col, ldb, stride_B, batch_count);
    hipblas_init<Tc>(hC, M, N, ldc, stride_C, batch_count);
    hC_gold = hC;

    // copy data from CPU to device
#ifdef __HIP_PLATFORM_NVCC__
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(Tb) * size_B, hipMemcpyHostToDevice));
#else
    if(std::is_same<Ta, int8_t>{} && transA == HIPBLAS_OP_N)
    {
        vector<Ta> hA_packed(hA);
        hipblas_packInt8(hA_packed, M, K, lda, batch_count, stride_A);
        CHECK_HIP_ERROR(
            hipMemcpy(dA, hA_packed.data(), sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }

    if(std::is_same<Tb, int8_t>{} && transB != HIPBLAS_OP_N)
    {
        vector<Tb> hB_packed(hB);
        hipblas_packInt8(hB_packed, N, K, ldb, batch_count, stride_B);
        CHECK_HIP_ERROR(
            hipMemcpy(dB, hB_packed.data(), sizeof(Tb) * size_B, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(Tb) * size_B, hipMemcpyHostToDevice));
    }
#endif
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(Tc) * size_C, hipMemcpyHostToDevice));

    status = hipblasGemmStridedBatchedExFn(handle,
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
                                           algo);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(Tc) * size_C, hipMemcpyDeviceToHost));

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

    // enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            unit_check_general<Tc>(
                M, N, ldc, hC_gold.data() + b * stride_C, hC.data() + b * stride_C);
        }
    }

    hipblasDestroy(handle);
    return status;
}

hipblasStatus_t testing_gemm_strided_batched_ex(const Arguments& argus)
{
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

    float alpha = argus.alpha;
    float beta  = argus.beta;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int norm_check = argus.norm_check;
    int unit_check = argus.unit_check;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf>(transA,
                                                                       transB,
                                                                       M,
                                                                       N,
                                                                       K,
                                                                       alpha,
                                                                       lda,
                                                                       ldb,
                                                                       beta,
                                                                       ldc,
                                                                       norm_check,
                                                                       unit_check,
                                                                       a_type,
                                                                       b_type,
                                                                       c_type,
                                                                       batch_count,
                                                                       compute_type,
                                                                       argus.fortran);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasHalf,
                                                          hipblasHalf,
                                                          hipblasHalf,
                                                          float>(transA,
                                                                 transB,
                                                                 M,
                                                                 N,
                                                                 K,
                                                                 alpha,
                                                                 lda,
                                                                 ldb,
                                                                 beta,
                                                                 ldc,
                                                                 norm_check,
                                                                 unit_check,
                                                                 a_type,
                                                                 b_type,
                                                                 c_type,
                                                                 batch_count,
                                                                 compute_type,
                                                                 argus.fortran);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && c_type == HIPBLAS_R_32F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_strided_batched_ex_template<float>(transA,
                                                                 transB,
                                                                 M,
                                                                 N,
                                                                 K,
                                                                 alpha,
                                                                 lda,
                                                                 ldb,
                                                                 beta,
                                                                 ldc,
                                                                 norm_check,
                                                                 unit_check,
                                                                 a_type,
                                                                 b_type,
                                                                 c_type,
                                                                 batch_count,
                                                                 compute_type,
                                                                 argus.fortran);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && c_type == HIPBLAS_R_64F && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_strided_batched_ex_template<double>(transA,
                                                                  transB,
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  alpha,
                                                                  lda,
                                                                  ldb,
                                                                  beta,
                                                                  ldc,
                                                                  norm_check,
                                                                  unit_check,
                                                                  a_type,
                                                                  b_type,
                                                                  c_type,
                                                                  batch_count,
                                                                  compute_type,
                                                                  argus.fortran);
    }
    else if(a_type == HIPBLAS_C_32F && b_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F
            && c_type == HIPBLAS_C_32F && compute_type == HIPBLAS_C_32F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasComplex>(transA,
                                                                          transB,
                                                                          M,
                                                                          N,
                                                                          K,
                                                                          alpha,
                                                                          lda,
                                                                          ldb,
                                                                          beta,
                                                                          ldc,
                                                                          norm_check,
                                                                          unit_check,
                                                                          a_type,
                                                                          b_type,
                                                                          c_type,
                                                                          batch_count,
                                                                          compute_type,
                                                                          argus.fortran);
    }
    else if(a_type == HIPBLAS_C_64F && b_type == HIPBLAS_C_64F && c_type == HIPBLAS_C_64F
            && c_type == HIPBLAS_C_64F && compute_type == HIPBLAS_C_64F)
    {
        status = testing_gemm_strided_batched_ex_template<hipblasDoubleComplex>(transA,
                                                                                transB,
                                                                                M,
                                                                                N,
                                                                                K,
                                                                                alpha,
                                                                                lda,
                                                                                ldb,
                                                                                beta,
                                                                                ldc,
                                                                                norm_check,
                                                                                unit_check,
                                                                                a_type,
                                                                                b_type,
                                                                                c_type,
                                                                                batch_count,
                                                                                compute_type,
                                                                                argus.fortran);
    }
    else if(a_type == HIPBLAS_R_8I && b_type == HIPBLAS_R_8I && c_type == HIPBLAS_R_32I
            && c_type == HIPBLAS_R_32I && compute_type == HIPBLAS_R_32I)
    {
        status = testing_gemm_strided_batched_ex_template<int8_t, int8_t, int32_t, int32_t>(
            transA,
            transB,
            M,
            N,
            K,
            alpha,
            lda,
            ldb,
            beta,
            ldc,
            norm_check,
            unit_check,
            a_type,
            b_type,
            c_type,
            batch_count,
            compute_type,
            argus.fortran);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
