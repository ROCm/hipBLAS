/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "arg_check.h"
#include "testing_common.hpp"
#include <typeinfo>

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_gemm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemmBatchedFn
        = FORTRAN ? hipblasGemmBatched<T, true> : hipblasGemmBatched<T, false>;

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

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

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    int batch_count = argus.batch_count;

    // bad arg checks
    if(batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0)
    {
        hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;
        hipblasLocalHandle handle(argus);

        const T *dA_array[1], *dB_array[1];
        T*       dC1_array[1];

        status = hipblasGemmBatchedFn(handle,
                                      transA,
                                      transB,
                                      M,
                                      N,
                                      K,
                                      &h_alpha,
                                      dA_array,
                                      lda,
                                      dB_array,
                                      ldb,
                                      &h_beta,
                                      dC1_array,
                                      ldc,
                                      batch_count);

        verify_hipblas_status_invalid_value(
            status,
            "ERROR: batch_count < 0 || M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 ");

        return status;
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

    if(lda < A_row || ldb < B_row || ldc < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    size_t A_size = size_t(lda) * A_col;
    size_t B_size = size_t(ldb) * B_col;
    size_t C_size = size_t(ldc) * N;

    // host arrays
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB(B_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_copy(C_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hB);
    hipblas_init(hC_host);

    hC_device.copy_from(hC_host);
    hC_copy.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // calculate "golden" result on CPU
        for(int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          h_alpha,
                          (T*)hA[i],
                          lda,
                          (T*)hB[i],
                          ldb,
                          h_beta,
                          (T*)hC_copy[i],
                          ldc);
        }

        // test hipBLAS batched gemm with alpha and beta pointers on device
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 d_alpha,
                                                 (const T* const*)dA.ptr_on_device(),
                                                 lda,
                                                 (const T* const*)dB.ptr_on_device(),
                                                 ldb,
                                                 d_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        // test hipBLAS batched gemm with alpha and beta pointers on host
        CHECK_HIP_ERROR(dC.transfer_from(hC_host));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                 transA,
                                                 transB,
                                                 M,
                                                 N,
                                                 K,
                                                 &h_alpha,
                                                 (const T* const*)dA.ptr_on_device(),
                                                 lda,
                                                 (const T* const*)dB.ptr_on_device(),
                                                 ldb,
                                                 &h_beta,
                                                 dC.ptr_on_device(),
                                                 ldc,
                                                 batch_count));

        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC_device);
        }

        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, hC_copy, hC_device, batch_count);
        }
    }

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

            CHECK_HIPBLAS_ERROR(hipblasGemmBatchedFn(handle,
                                                     transA,
                                                     transB,
                                                     M,
                                                     N,
                                                     K,
                                                     d_alpha,
                                                     (const T* const*)dA.ptr_on_device(),
                                                     lda,
                                                     (const T* const*)dB.ptr_on_device(),
                                                     ldb,
                                                     d_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
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
                      e_ldc,
                      e_batch_count>{}
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
