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
hipblasStatus_t testing_gbmv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasGbmvFn = FORTRAN ? hipblasGbmv<T, true> : hipblasGbmv<T, false>;

    int M    = argus.M;
    int N    = argus.N;
    int KL   = argus.KL;
    int KU   = argus.KU;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    if(transA == HIPBLAS_OP_N)
    {
        X_size = N;
        Y_size = M;
    }
    else
    {
        X_size = M;
        Y_size = N;
    }

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || KL < 0 || KU < 0 || lda < KL + KU + 1 || incx == 0 || incy == 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size * incx);
    host_vector<T> hy(Y_size * incy);
    host_vector<T> hy_host(Y_size * incy);
    host_vector<T> hy_device(Y_size * incy);
    host_vector<T> hy_cpu(Y_size * incy);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size * incx);
    device_vector<T> dy(Y_size * incy);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = (T)argus.alpha;
    T h_beta  = (T)argus.beta;

    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hx, 1, X_size, incx);
    hipblas_init<T>(hy, 1, Y_size, incy);

    // copy vector is easy in STL; hy_cpu = hy: save a copy in hy_cpu which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size * incx, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size * incy, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
        handle, transA, M, N, KL, KU, (T*)&h_alpha, dA, lda, dx, incx, (T*)&h_beta, dy, incy));

    CHECK_HIP_ERROR(
        hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size * incy, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size * incy, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(
        hipblasGbmvFn(handle, transA, M, N, KL, KU, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));

    CHECK_HIP_ERROR(
        hipMemcpy(hy_device.data(), dy, sizeof(T) * Y_size * incy, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_gbmv<T>(transA,
                      M,
                      N,
                      KL,
                      KU,
                      h_alpha,
                      hA.data(),
                      lda,
                      hx.data(),
                      incx,
                      h_beta,
                      hy_cpu.data(),
                      incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, incy, hy_cpu, hy_host);
            unit_check_general<T>(1, Y_size, incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, Y_size, incy, hy_cpu.data(), hy_host.data());
            hipblas_error_device
                = norm_check_general<T>('F', 1, Y_size, incy, hy_cpu.data(), hy_device.data());
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size * incy, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGbmvFn(
                handle, transA, M, N, KL, KU, d_alpha, dA, lda, dx, incx, d_beta, dy, incy));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_KL, e_KU, e_lda, e_incx, e_incy>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            gbmv_gflop_count<T>(transA, M, N, KL, KU),
            gbmv_gbyte_count<T>(transA, M, N, KL, KU),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
