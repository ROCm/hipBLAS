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
hipblasStatus_t testing_trtri_batched(Arguments argus)
{

    int N           = argus.N;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    int A_size = lda * N * batch_count;
    int bsa    = lda * N;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(lda < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(batch_count < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hB(A_size);
    vector<T> hA;

    // Initial Data on CPU
    srand(1);
    vector<T> hA_sub(bsa);
    for(size_t i = 0; i < batch_count; i++)
    {
        hipblas_init_symmetric<T>(hA_sub, N, lda);
        for(int j = 0; j < bsa; j++)
        {
            hA.push_back(hA_sub[j]);
        }
    }
    hB = hA;

    T *dA, *dinvA;

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    // char_uplo = 'U';
    hipblasFillMode_t uplo = char2hipblasFillMode_t(char_uplo);
    hipblasDiagType_t diag = char2hipblasDiagType_t(char_diag);

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dA, A_size * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dinvA, A_size * sizeof(T)));

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

    /* =====================================================================
           ROCBLAS
    =================================================================== */

    status = hipblasTrtri_batched<T>(
        handle, uplo, diag, N, dA, lda, bsa, dinvA, lda, bsa, batch_count);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dA));
        CHECK_HIP_ERROR(hipFree(dinvA));
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hA.data(), dinvA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(size_t i = 0; i < batch_count; i++)
        {
            int info = cblas_trtri<T>(char_uplo, char_diag, N, hB.data() + i * bsa, lda);
            if(info != 0)
                printf("error in cblas_trtri\n");
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N * batch_count, lda, hB.data(), hA.data());
        }

        for(int i = 0; i < 32; i++)
        {
            printf("CPU[%d]=%f, GPU[%d]=%f\n", i, hB[i], i, hA[i]);
        }
        // if enable norm check, norm check is invasive

    } // end of norm_check

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dinvA));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
