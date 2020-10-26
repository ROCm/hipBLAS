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

    int A_size, B_size, C_size, stride_A, stride_B, stride_C, A_row, A_col, B_row, B_col;
    int inc1_A, inc2_A, inc1_B, inc2_B;

    T hipblas_error = 0.0;

    hipblasStatus_t status1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status2 = HIPBLAS_STATUS_SUCCESS;

    if(transA == HIPBLAS_OP_N)
    {
        A_row  = M;
        A_col  = N;
        inc1_A = 1;
        inc2_A = lda;
    }
    else
    {
        A_row  = N;
        A_col  = M;
        inc1_A = lda;
        inc2_A = 1;
    }
    if(transB == HIPBLAS_OP_N)
    {
        B_row  = M;
        B_col  = N;
        inc1_B = 1;
        inc2_B = ldb;
    }
    else
    {
        B_row  = N;
        B_col  = M;
        inc1_B = ldb;
        inc2_B = 1;
    }

    stride_A = lda * A_col * stride_scale;
    stride_B = ldb * B_col * stride_scale;
    stride_C = ldc * N * stride_scale;

    A_size = stride_A * batch_count;
    B_size = stride_B * batch_count;
    C_size = stride_C * batch_count;

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    if(!dA || !dB || !dC || !d_alpha || !d_beta)
    {
        hipblasDestroy(handle);
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
         ROCBLAS
    =================================================================== */
    {
        // &h_alpha and &h_beta are host pointers
        status1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        if(status1 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status1;
        }

        status2 = hipblasGeamStridedBatchedFn(handle,
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
                                              batch_count);

        if(status2 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status2;
        }

        CHECK_HIP_ERROR(hipMemcpy(hC1.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
    }
    {
        // d_alpha and d_beta are device pointers
        status1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        if(status1 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status1;
        }

        CHECK_HIP_ERROR(hipMemcpy(dC, hC2.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

        status2 = hipblasGeamStridedBatchedFn(handle,
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
                                              batch_count);

        if(status2 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status2;
        }

        CHECK_HIP_ERROR(hipMemcpy(hC2.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));
    }

    /* =====================================================================
            CPU BLAS
    =================================================================== */
    if(status2 != HIPBLAS_STATUS_INVALID_VALUE) // only valid size compare with cblas
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
    }

    // enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(argus.unit_check)
    {
        unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC1);
        unit_check_general<T>(M, N, batch_count, ldc, stride_C, hC_copy, hC2);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
