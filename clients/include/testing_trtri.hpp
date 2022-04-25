/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trtri(const Arguments& argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasTrtriFn = FORTRAN ? hipblasTrtri<T, true> : hipblasTrtri<T, false>;

    const double rel_error = get_epsilon<T>() * 1000;

    int N = argus.N;
    int lda;
    int ldinvA = lda = argus.lda;

    size_t A_size = size_t(lda) * N;

    // check here to prevent undefined memory allocation error
    if(N < 0 || lda < 0 || lda < N)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hB(A_size);

    device_vector<T> dA(A_size);
    device_vector<T> dinvA(A_size);

    char char_uplo = argus.uplo_option;
    char char_diag = argus.diag_option;

    hipblasFillMode_t uplo = char2hipblas_fill(char_uplo);
    hipblasDiagType_t diag = char2hipblas_diagonal(char_diag);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init_symmetric<T>(hA, N, lda);

    // proprocess the matrix to avoid ill-conditioned matrix
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hA[i + j * lda] *= 0.01;

            if(j % 2)
                hA[i + j * lda] *= -1;
            if(uplo == HIPBLAS_FILL_MODE_LOWER && j > i)
                hA[i + j * lda] = 0.0f;
            else if(uplo == HIPBLAS_FILL_MODE_UPPER && j < i)
                hA[i + j * lda] = 0.0f;
            if(i == j)
            {
                if(diag == HIPBLAS_DIAG_UNIT)
                    hA[i + j * lda] = 1.0;
                else
                    hA[i + j * lda] *= 100.0;
            }
        }
    }

    hB = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dinvA, hA, sizeof(T) * A_size, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrtriFn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hA, dinvA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_trtri<T>(char_uplo, char_diag, N, hB, lda);

        if(argus.unit_check)
        {
            near_check_general<T>(N, N, lda, hB.data(), hA.data(), rel_error);
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', N, N, lda, hB, hA);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrtriFn(handle, uplo, diag, N, dA, lda, dinvA, ldinvA));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option, e_diag_option, e_N, e_lda>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            trtri_gflop_count<T>(N),
            trtri_gbyte_count<T>(N),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
