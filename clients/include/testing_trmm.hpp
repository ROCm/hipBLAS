/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trmm(Arguments argus)
{

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    char char_side   = argus.side_option;
    char char_uplo   = argus.uplo_option;
    char char_transA = argus.transA_option;
    char char_diag   = argus.diag_option;
    T    alpha       = argus.alpha;

    hipblasSideMode_t  side   = char2hipblasSideMode_t(char_side);
    hipblasFillMode_t  uplo   = char2hipblasFillMode_t(char_uplo);
    hipblasOperation_t transA = char2hipblas_operation(char_transA);
    hipblasDiagType_t  diag   = char2hipblasDiagType_t(char_diag);

    int K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    int A_size = lda * K;
    int B_size = ldb * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || lda < 0 || ldb < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hA(A_size);
    vector<T> hB(B_size);
    vector<T> hC(B_size);
    vector<T> hB_copy(B_size);

    T *dA, *dB, *dC;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error;

    hipblasHandle_t handle;

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dB, B_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dC, B_size * sizeof(T))); // dB and dC are exact the same size

    // Initial Data on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, K, lda);
    hipblas_init<T>(hB, M, N, ldb);
    hB_copy = hB;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * B_size, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    /*
        status = hipblasTrmm<T>(handle,
                side, uplo,
                transA, diag,
                M, N,
                &alpha,
                dA,lda,
                dB,ldb,
                dC,ldc);
    */

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dB));
        CHECK_HIP_ERROR(hipFree(dC));
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hC.data(), dC, sizeof(T) * B_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_trmm<T>(side, uplo, transA, diag, M, N, alpha, hA.data(), lda, hB_copy.data(), ldb);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, ldb, hB_copy.data(), hC.data());
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
