/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_geam_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGeamStridedBatchedFn
        = FORTRAN ? hipblasGeamStridedBatched<T, true> : hipblasGeamStridedBatched<T, false>;

    int M = argus.M;
    int N = argus.N;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int           A_row, A_col, B_row, B_col;
    hipblasStride stride_A, stride_B, stride_C;

    if(transA == HIPBLAS_OP_N)
    {
        A_row = M;
        A_col = N;
    }
    else
    {
        A_row = N;
        A_col = M;
    }
    if(transB == HIPBLAS_OP_N)
    {
        B_row = M;
        B_col = N;
    }
    else
    {
        B_row = N;
        B_col = M;
    }

    stride_A = size_t(lda) * A_col * stride_scale;
    stride_B = size_t(ldb) * B_col * stride_scale;
    stride_C = size_t(ldc) * N * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t B_size = stride_B * batch_count;
    size_t C_size = stride_C * batch_count;

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // allocate memory on device
    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(B_size);
    host_vector<T> hC1(C_size);
    host_vector<T> hC2(C_size);
    host_vector<T> hC_copy(C_size);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, A_row, A_col, lda, stride_A, batch_count);
    hipblas_init<T>(hB, B_row, B_col, ldb, stride_B, batch_count);
    hipblas_init<T>(hC1, M, N, ldc, stride_C, batch_count);

    hC2     = hC1;
    hC_copy = hC1;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC1.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    {
        // &h_alpha and &h_beta are host pointers
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        &h_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        &h_beta,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        dC,
                                                        ldc,
                                                        stride_C,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hC1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
    }
    {
        CHECK_HIP_ERROR(hipMemcpy(dC, hC2.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

        // d_alpha and d_beta are device pointers
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                        transA,
                                                        transB,
                                                        M,
                                                        N,
                                                        d_alpha,
                                                        dA,
                                                        lda,
                                                        stride_A,
                                                        d_beta,
                                                        dB,
                                                        ldb,
                                                        stride_B,
                                                        dC,
                                                        ldc,
                                                        stride_C,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hC2.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
    }

    /* =====================================================================
            CPU BLAS
    =================================================================== */
    if(argus.norm_check || argus.unit_check)
    {
        // reference calculation
        for(int b = 0; b < batch_count; b++)
        {
            cblas_geam(transA,
                       transB,
                       M,
                       N,
                       &h_alpha,
                       (T*)hA + b * stride_A,
                       lda,
                       &h_beta,
                       (T*)hB + b * stride_B,
                       ldb,
                       (T*)hC_copy + b * stride_C,
                       ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC1);
            unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC2);
        }

        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC1, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldc, stride_C, hC_copy, hC2, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasGeamStridedBatchedFn(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            d_beta,
                                                            dB,
                                                            ldb,
                                                            stride_B,
                                                            dC,
                                                            ldc,
                                                            stride_C,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_transA_option,
                      e_transB_option,
                      e_M,
                      e_N,
                      e_alpha,
                      e_lda,
                      e_stride_a,
                      e_beta,
                      e_ldb,
                      e_stride_b,
                      e_ldc,
                      e_stride_c,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         geam_gflop_count<T>(M, N),
                         geam_gbyte_count<T>(M, N),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
