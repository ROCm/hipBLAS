/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename Ta, typename Tb = Ta, typename Tc = Tb, typename Tex = Tc>
hipblasStatus_t testing_gemm_ex_template(const Arguments& argus)
{
    bool FORTRAN         = argus.fortran;
    auto hipblasGemmExFn = FORTRAN ? hipblasGemmExFortran : hipblasGemmEx;

    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    uint32_t          solution_index = 0;
    uint32_t          flags          = 0;
    size_t*           workspace_size = 0;
    void*             workspace      = 0;

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

    Tex h_alpha_Tc = argus.get_alpha<Tex>();
    Tex h_beta_Tc  = argus.get_beta<Tex>();

    int norm_check = argus.norm_check;
    int unit_check = argus.unit_check;
    int timing     = argus.timing;

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
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Ta>(hA, A_row, A_col, lda);
    hipblas_init_alternating_sign<Tb>(hB, B_row, B_col, ldb);
    hipblas_init<Tc>(hC_host, M, N, ldc);

    hC_gold = hC_device = hC_host;

    // copy data from CPU to device

    // CUDA doesn't do packing
#ifdef __HIP_PLATFORM_NVCC__
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(Tb) * size_B, hipMemcpyHostToDevice));
#else
    if(std::is_same<Ta, int8_t>{} && transA == HIPBLAS_OP_N && layout_pack_int8())
    {
        host_vector<Ta> hA_packed(hA);
        hipblas_packInt8(hA_packed, M, K, lda);
        CHECK_HIP_ERROR(hipMemcpy(dA, hA_packed, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }
    else
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(Ta) * size_A, hipMemcpyHostToDevice));
    }

    if(std::is_same<Tb, int8_t>{} && transB != HIPBLAS_OP_N && layout_pack_int8())
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
            unit_check_general<Tc>(M, N, ldc, hC_gold, hC_host);
            unit_check_general<Tc>(M, N, ldc, hC_gold, hC_device);
        }
        if(norm_check)
        {
            hipblas_error_host = std::abs(norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_host));
            hipblas_error_device
                = std::abs(norm_check_general<Tc>('F', M, N, ldc, hC_gold, hC_device));
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

        ArgumentModel<e_transA_option, e_transA_option, e_M, e_N, e_K, e_lda, e_ldb, e_ldc>{}
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

hipblasStatus_t testing_gemm_ex(const Arguments& argus)
{
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    hipblasDatatype_t a_type       = argus.a_type;
    hipblasDatatype_t b_type       = argus.b_type;
    hipblasDatatype_t c_type       = argus.c_type;
    hipblasDatatype_t compute_type = argus.compute_type;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_ex_template<hipblasHalf>(argus);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && c_type == HIPBLAS_R_16F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(a_type == HIPBLAS_R_16B && b_type == HIPBLAS_R_16B && c_type == HIPBLAS_R_16B
            && c_type == HIPBLAS_R_16B && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_ex_template<hipblasBfloat16, hipblasBfloat16, hipblasBfloat16, float>(
            argus);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && c_type == HIPBLAS_R_32F && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_ex_template<float>(argus);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && c_type == HIPBLAS_R_64F && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_ex_template<double>(argus);
    }
    else if(a_type == HIPBLAS_C_32F && b_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F
            && c_type == HIPBLAS_C_32F && compute_type == HIPBLAS_C_32F)
    {
        status = testing_gemm_ex_template<hipblasComplex>(argus);
    }
    else if(a_type == HIPBLAS_C_64F && b_type == HIPBLAS_C_64F && c_type == HIPBLAS_C_64F
            && c_type == HIPBLAS_C_64F && compute_type == HIPBLAS_C_64F)
    {
        status = testing_gemm_ex_template<hipblasDoubleComplex>(argus);
    }
    else if(a_type == HIPBLAS_R_8I && b_type == HIPBLAS_R_8I && c_type == HIPBLAS_R_32I
            && c_type == HIPBLAS_R_32I && compute_type == HIPBLAS_R_32I)
    {
        status = testing_gemm_ex_template<int8_t, int8_t, int32_t, int32_t>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
