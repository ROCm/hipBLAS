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
hipblasStatus_t testing_tbmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasTbmvFn = FORTRAN ? hipblasTbmv<T, true> : hipblasTbmv<T, false>;

    int M    = argus.M;
    int K    = argus.K;
    int lda  = argus.lda;
    int incx = argus.incx;

    int    abs_incx = incx >= 0 ? incx : -incx;
    size_t A_size   = size_t(lda) * M;
    size_t x_size   = size_t(M) * abs_incx;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || K < 0 || lda < K + 1 || !incx;
    if(invalid_size || !M)
    {
        hipblasStatus_t actual
            = hipblasTbmvFn(handle, uplo, transA, diag, M, K, nullptr, lda, nullptr, incx);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hx_cpu(x_size);
    host_vector<T> hx_res(x_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);

    double gpu_time_used, hipblas_error;

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, 1, A_size, 1);
    hipblas_init<T>(hx, 1, M, abs_incx);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hx_cpu = hx;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasTbmvFn(handle, uplo, transA, diag, M, K, dA, lda, dx, incx));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hx_res.data(), dx, sizeof(T) * x_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_tbmv<T>(uplo, transA, diag, M, K, hA.data(), lda, hx_cpu.data(), incx);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, abs_incx, hx_cpu, hx_res);
        }
        if(argus.norm_check)
        {
            hipblas_error
                = norm_check_general<T>('F', 1, M, abs_incx, hx_cpu.data(), hx_res.data());
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTbmvFn(handle, uplo, transA, diag, M, K, dA, lda, dx, incx));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option, e_transA_option, e_diag_option, e_M, e_K, e_lda, e_incx>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         tbmv_gflop_count<T>(M, K),
                         tbmv_gbyte_count<T>(M, K),
                         hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
