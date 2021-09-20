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

template <typename T, bool CONJ>
hipblasStatus_t testing_ger(const Arguments& argus)
{
    bool FORTRAN      = argus.fortran;
    auto hipblasGerFn = FORTRAN ? (CONJ ? hipblasGer<T, true, true> : hipblasGer<T, false, true>)
                                : (CONJ ? hipblasGer<T, true, false> : hipblasGer<T, false, false>);

    int M    = argus.M;
    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;
    int lda  = argus.lda;

    size_t A_size = size_t(lda) * N;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < 0 || incx <= 0 || incy <= 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_host(A_size);
    host_vector<T> hA_device(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hx(M * incx);
    host_vector<T> hy(N * incy);

    device_vector<T> dA(A_size);
    device_vector<T> dx(M * incx);
    device_vector<T> dy(N * incy);
    device_vector<T> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = argus.get_alpha<T>();

    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, N, incy);

    // copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * M * incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * N * incy, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGerFn(handle, M, N, (T*)&h_alpha, dx, incx, dy, incy, dA, lda));

        CHECK_HIP_ERROR(hipMemcpy(hA_host.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGerFn(handle, M, N, d_alpha, dx, incx, dy, incy, dA, lda));

        CHECK_HIP_ERROR(hipMemcpy(hA_device.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_ger<T, CONJ>(M, N, h_alpha, hx.data(), incx, hy.data(), incy, hA_cpu.data(), lda);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, lda, hA_cpu.data(), hA_host.data());
            unit_check_general<T>(M, N, lda, hA_cpu.data(), hA_device.data());
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, lda, hA_cpu.data(), hA_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', M, N, lda, hA_cpu.data(), hA_device.data());
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGerFn(handle, M, N, d_alpha, dx, incx, dy, incy, dA, lda));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_alpha, e_incx, e_incy, e_lda>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            ger_gflop_count<T>(M, N),
            ger_gbyte_count<T>(M, N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
