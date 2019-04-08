/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "arg_check.h"
#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "hipblas_unique_ptr.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename Td, typename Tc>
hipblasStatus_t testing_gemm_ex_template(hipblasOperation_t transA,
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
                                         hipblasDatatype_t  compute_type)
{
    hipblasGemmAlgo_t algo           = HIPBLAS_GEMM_DEFAULT;
    uint32_t          solution_index = 0;
    uint32_t          flags          = 0;
    size_t*           workspace_size = 0;
    void*             workspace      = 0;

    Td h_alpha_Td;
    Td h_beta_Td;

    if(is_same<Td, hipblasHalf>::value)
    {
        h_alpha_Td = float_to_half(alpha_float);
        h_beta_Td  = float_to_half(beta_float);
    }
    else if(is_same<Td, float>::value)
    {
        h_alpha_Td = static_cast<Td>(alpha_float);
        h_beta_Td  = static_cast<Td>(beta_float);
    }
    else if(is_same<Td, double>::value)
    {
        h_alpha_Td = static_cast<Td>(alpha_float);
        h_beta_Td  = static_cast<Td>(beta_float);
    }
    else
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    Tc h_alpha_Tc;
    Tc h_beta_Tc;

    if(is_same<Tc, hipblasHalf>::value)
    {
        h_alpha_Tc = float_to_half(alpha_float);
        h_beta_Tc  = float_to_half(beta_float);
    }
    else if(is_same<Tc, float>::value)
    {
        h_alpha_Tc = static_cast<Tc>(alpha_float);
        h_beta_Tc  = static_cast<Tc>(beta_float);
    }
    else if(is_same<Tc, double>::value)
    {
        h_alpha_Tc = static_cast<Tc>(alpha_float);
        h_beta_Tc  = static_cast<Tc>(beta_float);
    }
    else
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    hipblasHandle_t handle;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    hipblasCreate(&handle);

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

    // allocate memory on device
    //  auto dA_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_A),
    //                                       rocblas_test::device_free};
    //  auto dB_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_B),
    //                                       rocblas_test::device_free};
    //  auto dC_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_C),
    //                                       rocblas_test::device_free};
    //  auto dD_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Td) * size_D),
    //                                       rocblas_test::device_free};
    //  auto d_alpha_Tc_managed =
    //      rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Tc)), rocblas_test::device_free};
    //  auto d_beta_Tc_managed =
    //      rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(Tc)), rocblas_test::device_free};
    //  Td* dA         = (Td*)dA_managed.get();
    //  Td* dB         = (Td*)dB_managed.get();
    //  Td* dC         = (Td*)dC_managed.get();
    //  Td* dD         = (Td*)dD_managed.get();
    //  Tc* d_alpha_Tc = (Tc*)d_alpha_Tc_managed.get();
    //  Tc* d_beta_Tc  = (Tc*)d_beta_Tc_managed.get();

    Td *dA, *dB, *dC;
    Tc *d_alpha_Tc, *d_beta_Tc;

    CHECK_HIP_ERROR(hipMalloc(&dA, size_A * sizeof(Td)));
    CHECK_HIP_ERROR(hipMalloc(&dB, size_B * sizeof(Td)));
    CHECK_HIP_ERROR(hipMalloc(&dC, size_C * sizeof(Td)));

    CHECK_HIP_ERROR(hipMalloc(&d_alpha_Tc, sizeof(Td)));
    CHECK_HIP_ERROR(hipMalloc(&d_beta_Tc, sizeof(Td)));

    if(!dA || !dB || !dC || !d_alpha_Tc || !d_beta_Tc)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    vector<Td> hA(size_A);
    vector<Td> hB(size_B);
    vector<Td> hC(size_C);
    vector<Td> hC_gold(size_C);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Td>(hA, A_row, A_col, lda);
    hipblas_init_alternating_sign<Td>(hB, B_row, B_col, ldb);
    hipblas_init<Td>(hC, M, N, ldc);

    //  if(is_same<Td, hipblasHalf>::value)
    //  {
    //      std::cout << "----A-----------------" << std::endl;
    //      for(int i = 0; i < size_A; i++){ cout << half_to_float(hA[i]) << "  "; }
    //      std::cout << std::endl << "-----B-----------------" << std::endl;
    //      for(int i = 0; i < size_B; i++){ cout << half_to_float(hB[i]) << "  "; }
    //      std::cout << std::endl << "-----C-----------------" << std::endl;
    //      for(int i = 0; i < size_C; i++){ cout << half_to_float(hC[i]) << "  "; }
    //      std::cout << std::endl << "-----D-----------------" << std::endl;
    //      for(int i = 0; i < size_D; i++){ cout << half_to_float(hD_1[i]) << "  "; }
    //      std::cout << std::endl << "-----------------------" << std::endl;
    //  }
    //  else
    //  {
    //      std::cout << "----A-----------------" << std::endl;
    //      for(int i = 0; i < size_A; i++){ cout << hA[i] << "  "; }
    //      std::cout << std::endl << "-----B-----------------" << std::endl;
    //      for(int i = 0; i < size_B; i++){ cout << hB[i] << "  "; }
    //      std::cout << std::endl << "-----C-----------------" << std::endl;
    //      for(int i = 0; i < size_C; i++){ cout << hC[i] << "  "; }
    //      std::cout << std::endl << "-----D-----------------" << std::endl;
    //      for(int i = 0; i < size_D; i++){ cout << hD_1[i] << "  "; }
    //      std::cout << std::endl << "-----------------------" << std::endl;
    //  }

    hC_gold = hC;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(Td) * size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(Td) * size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(Td) * size_C, hipMemcpyHostToDevice));

    status = hipblasGemmEx(handle,
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
                           algo);

    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(Td) * size_C, hipMemcpyDeviceToHost));

    //      std::cout << std::endl << "-----hD_1---------------------------------------" <<
    //      std::endl;
    //      if(is_same<Td, hipblasHalf>::value)
    //      {
    //                  for(int i = 0; i < size_D; i++){ cout << half_to_float(hD_1[i]) << "  ";
    //                  }
    //      }
    //      else
    //      {
    //                  for(int i = 0; i < size_D; i++){ cout << hD_1[i] << "  "; }
    //      }
    //      std::cout << std::endl << "------------------------------------------------" <<
    //      std::endl;

    // CPU BLAS

    cblas_gemm<Td>(transA,
                   transB,
                   M,
                   N,
                   K,
                   h_alpha_Td,
                   hA.data(),
                   lda,
                   hB.data(),
                   ldb,
                   h_beta_Td,
                   hC_gold.data(),
                   ldc);

//      std::cout << std::endl << "---gold---gold---gold---------------------" << std::endl;
//      if(is_same<Td, hipblasHalf>::value)
//      {
//          for(int i = 0; i < size_D; i++){ std::cout << half_to_float(hD_gold[i]) << "  "; }
//      }
//      else
//      {
//          for(int i = 0; i < size_D; i++){ std::cout << hD_gold[i] << "  "; }
//      }
//      std::cout << std::endl << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;

#ifndef NDEBUG
// print_matrix(hC_gold, hC, min(M, 3), min(N, 3), ldc);
#endif

    // enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(unit_check)
    {
        unit_check_general<Td>(M, N, ldc, hC_gold.data(), hC.data());
    }

    hipblasDestroy(handle);
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));

    CHECK_HIP_ERROR(hipFree(d_alpha_Tc));
    CHECK_HIP_ERROR(hipFree(d_beta_Tc));

    return status;
}

hipblasStatus_t testing_gemm_ex(Arguments argus)
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

    float alpha = argus.alpha;
    float beta  = argus.beta;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int norm_check = argus.norm_check;
    int unit_check = argus.unit_check;

    if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
       && c_type == HIPBLAS_R_16F
       && compute_type == HIPBLAS_R_16F)
    {
        status = testing_gemm_ex_template<hipblasHalf, hipblasHalf>(transA,
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
                                                                    compute_type);
    }
    else if(a_type == HIPBLAS_R_16F && b_type == HIPBLAS_R_16F && c_type == HIPBLAS_R_16F
            && c_type == HIPBLAS_R_16F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_ex_template<hipblasHalf, float>(transA,
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
                                                              compute_type);
    }
    else if(a_type == HIPBLAS_R_32F && b_type == HIPBLAS_R_32F && c_type == HIPBLAS_R_32F
            && c_type == HIPBLAS_R_32F
            && compute_type == HIPBLAS_R_32F)
    {
        status = testing_gemm_ex_template<float, float>(transA,
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
                                                        compute_type);
    }
    else if(a_type == HIPBLAS_R_64F && b_type == HIPBLAS_R_64F && c_type == HIPBLAS_R_64F
            && c_type == HIPBLAS_R_64F
            && compute_type == HIPBLAS_R_64F)
    {
        status = testing_gemm_ex_template<double, double>(transA,
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
                                                          compute_type);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
