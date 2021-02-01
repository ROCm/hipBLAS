/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_trsv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTrsvFn = FORTRAN ? hipblasTrsv<T, true> : hipblasTrsv<T, false>;

    int                M           = argus.M;
    int                incx        = argus.incx;
    int                lda         = argus.lda;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = lda * M;
    int size_x   = abs_incx * M;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(lda < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incx == 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);
    host_vector<T> hx_or_b_2(size_x);
    host_vector<T> cpu_x_or_b(size_x);

    device_vector<T> dA(size_A);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double hipblas_error;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda);

    //  calculate AAT = hA * hA ^ T
    cblas_gemm<T>(HIPBLAS_OP_N,
                  HIPBLAS_OP_T,
                  M,
                  M,
                  M,
                  (T)1.0,
                  hA.data(),
                  lda,
                  hA.data(),
                  lda,
                  (T)0.0,
                  AAT.data(),
                  lda);

    //  copy AAT into hA, make hA strictly diagonal dominant, and therefore SPD
    for(int i = 0; i < M; i++)
    {
        T t = 0.0;
        for(int j = 0; j < M; j++)
        {
            hA[i + j * lda] = AAT[i + j * lda];
            t += std::abs(AAT[i + j * lda]);
        }
        hA[i + i * lda] = t;
    }
    //  calculate Cholesky factorization of SPD matrix hA
    cblas_potrf<T>(char_uplo, M, hA.data(), lda);

    //  make hA unit diagonal if diag == rocblas_diagonal_unit
    if(char_diag == 'U' || char_diag == 'u')
    {
        if('L' == char_uplo || 'l' == char_uplo)
            for(int i = 0; i < M; i++)
            {
                T diag = hA[i + i * lda];
                for(int j = 0; j <= i; j++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
        else
            for(int j = 0; j < M; j++)
            {
                T diag = hA[j + j * lda];
                for(int i = 0; i <= j; i++)
                    hA[i + j * lda] = hA[i + j * lda] / diag;
            }
    }

    hipblas_init<T>(hx, 1, M, abs_incx);
    hb = hx;

    // Calculate hb = hA*hx;
    cblas_trmv<T>(uplo, transA, diag, M, hA.data(), lda, hb.data(), incx);
    cpu_x_or_b = hb; // cpuXorB <- B
    hx_or_b_1  = hb;
    hx_or_b_2  = hb;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * size_A, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        status = hipblasTrsvFn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        // copy output from device to CPU
        hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost);

        // Calculating error
        hipblas_error = std::abs(vector_norm_1<T>(M, abs_incx, hx.data(), hx_or_b_1.data()));

        if(argus.unit_check)
        {
            double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * M;
            unit_check_error(hipblas_error, tolerance);
        }
    }

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

            status = hipblasTrsvFn(handle, uplo, transA, diag, M, dA, lda, dx_or_b, incx);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_M, e_lda, e_incx>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trsv_gflop_count<T>(M),
                         trsv_gbyte_count<T>(M),
                         hipblas_error);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
