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
hipblasStatus_t testing_tbsv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTbsvFn = FORTRAN ? hipblasTbsv<T, true> : hipblasTbsv<T, false>;

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

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || K < 0 || lda < K + 1 || !incx)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    int abs_incx = incx < 0 ? -incx : incx;
    int size_A   = size_t(M) * M;
    int size_AB  = size_t(lda) * M;
    int size_x   = abs_incx * M;

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
    hipblas_init<T>(hA, M, M, M);

    banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA, M, M, K);

    prepare_triangular_solve((T*)hA, M, (T*)AAT, M, char_uplo);
    if(diag == HIPBLAS_DIAG_UNIT)
    {
        make_unit_diagonal(uplo, (T*)hA, M, M);
    }

    regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA, M, (T*)hAB, lda, M, K);
    hipMemcpy(dAB, hAB.data(), sizeof(T) * size_AB, hipMemcpyHostToDevice);

    hipblas_init<T>(hx, 1, M, abs_incx);
    hb = hx;

    cblas_tbmv<T>(uplo, transA, diag, M, K, hAB, lda, hb, incx);
    hx_or_b_1 = hb;

    // copy data from CPU to device
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
        status = hipblasTbsvFn(handle, uplo, transA, diag, M, K, dAB, lda, dx_or_b, incx);

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
        real_t<T> eps       = std::numeric_limits<real_t<T>>::epsilon();
        double    tolerance = eps * 40 * M;

        double error = 0.0;
        if(argus.unit_check)
        {
            double max_err_scal = 0.0, max_err = 0.0;
            for(int i = 0; i < M; i++)
            {
                T diff = (hx[i * abs_incx] - hx_or_b_1[i * abs_incx]);
                if(diff != T(0))
                {
                    max_err += abs(diff);
                }
                max_err_scal += abs(hx_or_b_1[i * abs_incx]);
            }
            error = max_err / max_err_scal;
            unit_check_error(error, tolerance);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
