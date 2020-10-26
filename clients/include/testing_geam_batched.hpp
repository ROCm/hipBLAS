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
hipblasStatus_t testing_geam_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGeamBatchedFn
        = FORTRAN ? hipblasGeamBatched<T, true> : hipblasGeamBatched<T, false>;

    int M = argus.M;
    int N = argus.N;

    int lda = argus.lda;
    int ldb = argus.ldb;
    int ldc = argus.ldc;

    int batch_count = argus.batch_count;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    int A_size, B_size, C_size, A_row, A_col, B_row, B_col;
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

    A_size = lda * A_col;
    B_size = ldb * B_col;
    C_size = ldc * N;

    // check here to prevent undefined memory allocation error
    if(M <= 0 || N <= 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    device_batch_vector<T> bA_array(batch_count, A_size);
    device_batch_vector<T> bB_array(batch_count, B_size);
    device_batch_vector<T> bC_array(batch_count, C_size);

    device_vector<T*, 0, T> dA(batch_count);
    device_vector<T*, 0, T> dB(batch_count);
    device_vector<T*, 0, T> dC(batch_count);
    device_vector<T>        d_alpha(1);
    device_vector<T>        d_beta(1);

    int last = batch_count - 1;
    if(!dA || !dB || !dC || !d_alpha || !d_beta || (!bA_array[last] && A_size)
       || (!bB_array[last] && B_size) || (!bC_array[last] && C_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hB[batch_count];
    host_vector<T> hC1[batch_count];
    host_vector<T> hC2[batch_count];
    host_vector<T> hC_copy[batch_count];

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]      = host_vector<T>(A_size);
        hB[b]      = host_vector<T>(B_size);
        hC1[b]     = host_vector<T>(C_size);
        hC2[b]     = host_vector<T>(C_size);
        hC_copy[b] = host_vector<T>(C_size);

        hipblas_init<T>(hA[b], A_row, A_col, lda);
        hipblas_init<T>(hB[b], B_row, B_col, ldb);
        hipblas_init<T>(hC1[b], M, N, ldc);
        hC2[b]     = hC1[b];
        hC_copy[b] = hC1[b];

        CHECK_HIP_ERROR(hipMemcpy(bA_array[b], hA[b], sizeof(T) * A_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bB_array[b], hB[b], sizeof(T) * B_size, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(bC_array[b], hC1[b], sizeof(T) * C_size, hipMemcpyHostToDevice));
    }

    CHECK_HIP_ERROR(hipMemcpy(dA, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, bB_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, bC_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));
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

        status2 = hipblasGeamBatchedFn(handle,
                                       transA,
                                       transB,
                                       M,
                                       N,
                                       &h_alpha,
                                       dA,
                                       lda,
                                       &h_beta,
                                       dB,
                                       ldb,
                                       dC,
                                       ldc,
                                       batch_count);

        if(status2 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status2;
        }

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hC1[b], bC_array[b], sizeof(T) * C_size, hipMemcpyDeviceToHost));
        }
    }
    {
        // d_alpha and d_beta are device pointers
        status1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        if(status1 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status1;
        }

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(bC_array[b], hC2[b], sizeof(T) * C_size, hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(dC, bC_array, batch_count * sizeof(T*), hipMemcpyHostToDevice));

        status2 = hipblasGeamBatchedFn(
            handle, transA, transB, M, N, d_alpha, dA, lda, d_beta, dB, ldb, dC, ldc, batch_count);

        if(status2 != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status2;
        }

        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(
                hipMemcpy(hC2[b], bC_array[b], sizeof(T) * C_size, hipMemcpyDeviceToHost));
        }
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
                       (T*)hA[b],
                       lda,
                       &h_beta,
                       (T*)hB[b],
                       ldb,
                       (T*)hC_copy[b],
                       ldc);
        }
    }

    // enable unit check, notice unit check is not invasive, but norm check is,
    // unit check and norm check can not be interchanged their order
    if(argus.unit_check)
    {
        unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC1);
        unit_check_general<T>(M, N, batch_count, ldc, hC_copy, hC2);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
