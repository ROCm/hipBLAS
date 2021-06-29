/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_gemm_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemmStridedBatchedFn
        = FORTRAN ? hipblasGemmStridedBatched<T, true> : hipblasGemmStridedBatched<T, false>;

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int    lda          = argus.lda;
    int    ldb          = argus.ldb;
    int    ldc          = argus.ldc;
    int    batch_count  = argus.batch_count;
    double stride_scale = argus.stride_scale;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    int A_row, A_col, B_row, B_col;

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

    hipblasStride stride_A = size_t(lda) * A_col * stride_scale;
    hipblasStride stride_B = size_t(ldb) * B_col * stride_scale;
    hipblasStride stride_C = size_t(ldc) * N * stride_scale;
    size_t        A_size   = stride_A * batch_count;
    size_t        B_size   = stride_B * batch_count;
    size_t        C_size   = stride_C * batch_count;

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
    hipblas_init<T>(hA, A_row, A_col * batch_count, lda);
    hipblas_init<T>(hB, B_row, B_col * batch_count, ldb);
    hipblas_init<T>(hC_host, M, N * batch_count, ldc);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy   = hC_host;
    hC_device = hC_host;

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC_host, sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        // host mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        // library interface
        CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        &h_beta,
                                                        dC,
                                                        ldc,
                                                        stride_C,
                                                        batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hC_host, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        // device mode
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_device, sizeof(T) * C_size, hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        K,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        d_beta,
                                                        dC,
                                                        ldc,
                                                        stride_C,
                                                        batch_count));
        CHECK_HIP_ERROR(hipMemcpy(hC_device, dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int i = 0; i < batch_count; i++)
        {
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          h_alpha,
                          hA.data() + stride_A * i,
                          lda,
                          hB.data() + stride_B * i,
                          ldb,
                          h_beta,
                          hC_copy.data() + stride_C * i,
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC_host);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasGemmStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_C,
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
                      e_stride_a,
                      e_ldb,
                      e_stride_b,
                      e_beta,
                      e_ldc,
                      e_stride_c,
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
