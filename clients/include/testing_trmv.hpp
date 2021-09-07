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
hipblasStatus_t testing_trmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTrmvFn = FORTRAN ? hipblasTrmv<T, true> : hipblasTrmv<T, false>;

    int M    = argus.M;
    int lda  = argus.lda;
    int incx = argus.incx;

    size_t A_size = size_t(lda) * M;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || lda < M || incx == 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(M * incx);
    host_vector<T> hres(M * incx);

    device_vector<T> dA(A_size);
    device_vector<T> dx(M * incx);

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda);
    hipblas_init<T>(hx, 1, M, incx);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hres = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * lda * M, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * M * incx, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTrmvFn(handle, uplo, transA, diag, M, dA, lda, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hres.data(), dx, sizeof(T) * M * incx, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_trmv<T>(uplo, transA, diag, M, hA.data(), lda, hx.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, incx, hx, hres);
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, incx, hx.data(), hres.data());
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

            CHECK_HIPBLAS_ERROR(hipblasTrmvFn(handle, uplo, transA, diag, M, dA, lda, dx, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_M, e_lda, e_incx>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmv_gflop_count<T>(M),
                         trmv_gbyte_count<T>(M),
                         hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
