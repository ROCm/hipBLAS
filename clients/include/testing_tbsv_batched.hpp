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
hipblasStatus_t testing_tbsv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTbsvBatchedFn
        = FORTRAN ? hipblasTbsvBatched<T, true> : hipblasTbsvBatched<T, false>;

    int                M           = argus.M;
    int                K           = argus.K;
    int                incx        = argus.incx;
    int                lda         = argus.lda;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = argus.batch_count;

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = size_t(M) * M;
    int size_AB  = size_t(lda) * M;
    int size_x   = abs_incx * M;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || K < 0 || lda < K + 1 || incx == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA[batch_count];
    host_vector<T> hAB[batch_count];
    host_vector<T> AAT[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<T> hx[batch_count];
    host_vector<T> hx_or_b_1[batch_count];

    device_batch_vector<T> bAB(batch_count, size_AB);
    device_batch_vector<T> bx_or_b(batch_count, size_x);

    device_vector<T*, 0, T> dAB(batch_count);
    device_vector<T*, 0, T> dx_or_b(batch_count);

    int last = batch_count - 1;
    if(!dAB || !dx_or_b || (!bAB[last] && size_AB) || (!bx_or_b[last] && size_x))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA[b]        = host_vector<T>(size_A);
        hAB[b]       = host_vector<T>(size_AB);
        AAT[b]       = host_vector<T>(size_A);
        hb[b]        = host_vector<T>(size_x);
        hx[b]        = host_vector<T>(size_x);
        hx_or_b_1[b] = host_vector<T>(size_x);

        srand(1);
        hipblas_init<T>(hA[b], M, M, M);

        banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], M, M, K);

        prepare_triangular_solve((T*)hA[b], M, (T*)AAT[b], M, char_uplo);
        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, (T*)hA[b], M, M);
        }

        regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], M, (T*)hAB[b], lda, M, K);

        hipblas_init<T>(hx[b], 1, M, abs_incx);
        hb[b] = hx[b];

        // Calculate hb = hA*hx;
        cblas_tbmv<T>(uplo, transA, diag, M, K, hAB[b], lda, hb[b], incx);
        hx_or_b_1[b] = hb[b];

        // copy data from CPU to device
        hipMemcpy(bAB[b], hAB[b], sizeof(T) * size_AB, hipMemcpyHostToDevice);
        hipMemcpy(bx_or_b[b], hx_or_b_1[b], sizeof(T) * size_x, hipMemcpyHostToDevice);
    }
    hipMemcpy(dAB, bAB, sizeof(T*) * batch_count, hipMemcpyHostToDevice);
    hipMemcpy(dx_or_b, bx_or_b, sizeof(T*) * batch_count, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(argus.timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasTbsvBatchedFn(
            handle, uplo, transA, diag, M, K, dAB, lda, dx_or_b, incx, batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hx_or_b_1[b], bx_or_b[b], sizeof(T) * size_x, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
            double    tolerance = eps * 40 * M;

            double error = 0.0, max_err = 0.0, max_err_scal = 0.0;
            for(int i = 0; i < M; i++)
            {
                T diff = (hx[b][i * abs_incx] - hx_or_b_1[b][i * abs_incx]);
                if(diff != T(0))
                {
                    max_err += abs(diff);
                }
                max_err_scal += abs(hx_or_b_1[b][i * abs_incx]);
            }
            error = max_err / max_err_scal;
            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
