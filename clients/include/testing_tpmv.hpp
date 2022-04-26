/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_tpmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTpmvFn = FORTRAN ? hipblasTpmv<T, true> : hipblasTpmv<T, false>;

    int M    = argus.M;
    int incx = argus.incx;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t x_size   = size_t(M) * abs_incx;
    size_t A_size   = size_t(M) * (M + 1) / 2;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || !incx;
    if(invalid_size || !M)
    {
        hipblasStatus_t actual
            = hipblasTpmvFn(handle, uplo, transA, diag, M, nullptr, nullptr, incx);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hres(x_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    hipblas_init_matrix(hA, argus, A_size, 1, 1, 0, 1, hipblas_client_never_set_nan, true, false);
    hipblas_init_vector(hx, argus, M, abs_incx, 0, 1, hipblas_client_never_set_nan, false, true);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hres = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTpmvFn(handle, uplo, transA, diag, M, dA, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hres.data(), dx, sizeof(T) * x_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_tpmv<T>(uplo, transA, diag, M, hA.data(), hx.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, abs_incx, hx, hres);
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, abs_incx, hx.data(), hres.data());
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

            CHECK_HIPBLAS_ERROR(hipblasTpmvFn(handle, uplo, transA, diag, M, dA, dx, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_M, e_incx>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            tpmv_gflop_count<T>(M),
            tpmv_gbyte_count<T>(M),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
