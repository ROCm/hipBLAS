/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
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
hipblasStatus_t testing_tbsv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTbsvStridedBatchedFn
        = FORTRAN ? hipblasTbsvStridedBatched<T, true> : hipblasTbsvStridedBatched<T, false>;

    int                M            = argus.M;
    int                K            = argus.K;
    int                incx         = argus.incx;
    int                lda          = argus.lda;
    char               char_uplo    = argus.uplo_option;
    char               char_diag    = argus.diag_option;
    char               char_transA  = argus.transA_option;
    hipblasFillMode_t  uplo         = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag         = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA       = char2hipblas_operation(char_transA);
    double             stride_scale = argus.stride_scale;
    int                batch_count  = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int strideA  = M * M;
    int strideAB = M * lda * stride_scale;
    int stridex  = abs_incx * M * stride_scale;
    int size_A   = strideA * batch_count;
    int size_AB  = strideAB * batch_count;
    int size_x   = stridex * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || K < 0 || lda < K + 1 || !incx || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count || !M || !lda)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(size_A);
    host_vector<T> hAB(size_AB);
    host_vector<T> AAT(size_A);
    host_vector<T> hb(size_x);
    host_vector<T> hx(size_x);
    host_vector<T> hx_or_b_1(size_x);

    device_vector<T> dAB(size_AB);
    device_vector<T> dx_or_b(size_x);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, M, strideA, batch_count);
    hipblas_init<T>(hx, 1, M, abs_incx, stridex, batch_count);
    hb = hx;

    for(int b = 0; b < batch_count; b++)
    {
        T* hAbat  = hA.data() + b * strideA;
        T* hABbat = hAB.data() + b * strideAB;
        T* AATbat = AAT.data() + b * strideA;
        T* hbbat  = hb.data() + b * stridex;
        banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, hAbat, M, M, K);

        prepare_triangular_solve(hAbat, M, AATbat, M, char_uplo);
        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, hAbat, M, M);
        }

        regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, hAbat, M, hABbat, lda, M, K);

        // Calculate hb = hA*hx;
        cblas_tbmv<T>(uplo, transA, diag, M, K, hABbat, lda, hbbat, incx);
    }

    hx_or_b_1 = hb;

    // copy data from CPU to device
    hipMemcpy(dAB, hAB.data(), sizeof(T) * size_AB, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, hx_or_b_1.data(), sizeof(T) * size_x, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasTbsvStridedBatchedFn(handle,
                                             uplo,
                                             transA,
                                             diag,
                                             M,
                                             K,
                                             dAB,
                                             lda,
                                             strideAB,
                                             dx_or_b,
                                             incx,
                                             stridex,
                                             batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hx_or_b_1.data(), dx_or_b, sizeof(T) * size_x, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
            double    tolerance = eps * 40 * M;

            T*     hxb          = hx + b * stridex;
            T*     hx_or_b_1b   = hx_or_b_1 + b * stridex;
            double max_err_scal = 0.0, max_err = 0.0, error = 0.0;
            for(int i = 0; i < M; i++)
            {
                T diff = (hxb[i * abs_incx] - hx_or_b_1b[i * abs_incx]);
                if(diff != T(0))
                {
                    max_err += abs(diff);
                }
                max_err_scal += abs(hx_or_b_1b[i * abs_incx]);
            }

            error = max_err / max_err_scal;
            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
