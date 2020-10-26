/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <typeinfo>
#include <vector>

#include "hipblas_unique_ptr.hpp"
#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_GemmStridedBatched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemmStridedBatchedFn
        = FORTRAN ? hipblasGemmStridedBatched<T, true> : hipblasGemmStridedBatched<T, false>;

    int M = argus.M;
    int N = argus.N;
    int K = argus.K;

    int lda         = argus.lda;
    int ldb         = argus.ldb;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < 0 || ldb < 0 || ldc < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasOperation_t transB = char2hipblas_operation(argus.transB_option);

    int A_size, B_size, C_size, A_row, A_col, B_row, B_col;
    int bsa, bsb, bsc; // batch size A, B, C
    T   alpha = argus.alpha;
    T   beta  = argus.beta;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;

    T               rocblas_error = 0.0;
    hipblasHandle_t handle;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    hipblasCreate(&handle);

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

    bsa    = lda * A_col * 2;
    bsb    = ldb * B_col * 2;
    bsc    = ldc * N;
    A_size = bsa * batch_count;
    B_size = bsb * batch_count;
    C_size = bsc * batch_count;

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
    hipblas_init<T>(hA, A_row, A_col * batch_count, lda);
    hipblas_init<T>(hB, B_row, B_col * batch_count, ldb);
    hipblas_init<T>(hC, M, N * batch_count, ldc);

    // copy vector is easy in STL; hz = hx: save a copy in hC_copy which will be output of CPU BLAS
    hC_copy = hC;

    // copy data from CPU to device, does not work for lda != A_row
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC.data(), sizeof(T) * C_size, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */

    // library interface
    status = hipblasGemmStridedBatchedFn(handle,
                                         transA,
                                         transB,
                                         M,
                                         N,
                                         K,
                                         &alpha,
                                         dA,
                                         lda,
                                         bsa,
                                         dB,
                                         ldb,
                                         bsb,
                                         &beta,
                                         dC,
                                         ldc,
                                         bsc,
                                         batch_count);

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T) * C_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {

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
                          alpha,
                          hA.data() + bsa * i,
                          lda,
                          hB.data() + bsb * i,
                          ldb,
                          beta,
                          hC_copy.data() + bsc * i,
                          ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N * batch_count, lda, hC_copy.data(), hC.data());
        }

    } // end of if unit/norm check

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
