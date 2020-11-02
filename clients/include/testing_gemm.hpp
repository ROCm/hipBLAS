/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
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

    T alpha = argus.alpha;
    T beta  = argus.beta;

    int A_size, B_size, C_size, A_row, A_col, B_row, B_col;

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

    A_size = lda * A_col;
    B_size = ldb * B_col;
    C_size = ldc * N;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    hipblasCreate(&handle);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(C_size);
    vector<T> hC_copy(C_size);

    device_vector<T> dA(A_size);
    device_vector<T> dB(B_size);
    device_vector<T> dC(C_size);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, A_row, A_col, lda);
    hipblas_init<T>(hB, B_row, B_col, ldb);
    hipblas_init<T>(hC, M, N, ldc);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy = hC;

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * lda * A_col, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * ldb * B_col, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * ldc * N, hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // library interface
    status
        = hipblasGemmFn(handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T) * ldc * N, hipMemcpyDeviceToHost));

    double hipblas_error = 0.0;
    double gpu_time_used = 0.0;
    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        if(status != HIPBLAS_STATUS_INVALID_VALUE)
        { // only valid size compare with cblas
            cblas_gemm<T>(transA,
                          transB,
                          M,
                          N,
                          K,
                          alpha,
                          hA.data(),
                          lda,
                          hB.data(),
                          ldb,
                          beta,
                          hC_copy.data(),
                          ldc);
        }

#ifndef NDEBUG
        print_matrix(hC_copy, hC, min(M, 3), min(N, 3), ldc);
#endif

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldc, hC_copy.data(), hC.data());
        }
        if(argus.norm_check)
        {
            hipblas_error
                = std::abs(norm_check_general<T>('F', M, N, ldc, hC_copy.data(), hC.data()));
        }

    } // end of if unit/norm check

    if(argus.timing)
    {
        hipStream_t stream;
        status = hipblasGetStream(handle, &stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            status = hipblasGemmFn(
                handle, transA, transB, M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
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
                         hipblas_error);
    }

    hipblasDestroy(handle);
    return status;
}
