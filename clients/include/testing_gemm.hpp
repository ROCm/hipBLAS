/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_gemm(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasGemmFn = FORTRAN ? hipblasGemm<T, true> : hipblasGemm<T, false>;

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    float alpha_float = argus.alpha;
    float beta_float  = argus.beta;

    T h_alpha, h_beta;

    if(is_same<T, hipblasHalf>::value)
    {
        h_alpha = float_to_half(alpha_float);
        h_beta  = float_to_half(beta_float);
    }
    else
    {
        h_alpha = static_cast<T>(alpha_float);
        h_beta  = static_cast<T>(beta_float);
    }

    int A_row, A_col, B_row, B_col;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = K;
    }
    else
    {
        A_row = K;
        A_col = M;
    }

    if(transB == HIPBLAS_OP_N)
    {
        B_row = K;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = K;
    }

    size_t A_size = size_t(lda) * A_col;
    size_t B_size = size_t(ldb) * B_col;
    size_t C_size = size_t(ldc) * N;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC_host(C_size);
    host_vector<T> hC_device(C_size);
    host_vector<T> hC_copy(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, A_row, A_col, lda);
    hipblas_init<T>(hB, B_row, B_col, ldb);
    hipblas_init<T>(hC_host, M, N, ldc);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy   = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * lda * A_col, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * ldb * B_col, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * ldc * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

    // library interface
    CHECK_HIPBLAS_ERROR(hipblasGemmFn(
        handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * ldc * N, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * ldc * N, hipMemcpyHostToDevice));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(
        hipblasGemmFn(handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
    CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * ldc * N, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_gemm<T>(transA,
                      transB,
                      M,
                      N,
                      K,
                      h_alpha,
                      hA.data(),
                      lda,
                      hB.data(),
                      ldb,
                      h_beta,
                      hC_copy.data(),
                      ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_copy, hC_host);
            unit_check_general<T>(M, N, ldc, hC_copy, hC_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host = std::abs(norm_check_general<T>('F', M, N, ldc, hC_copy, hC_host));
            hipblas_error_device
                = std::abs(norm_check_general<T>('F', M, N, ldc, hC_copy, hC_device));
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGemmFn(
                handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA_option,
                      e_transB_option,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_ldb,
                      e_beta,
                      e_ldc>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         gemm_gflop_count<T>(M, N, K),
                         gemm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
