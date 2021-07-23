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
hipblasStatus_t testing_hemv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHemvBatchedFn
        = FORTRAN ? hipblasHemvBatched<T, true> : hipblasHemvBatched<T, false>;

    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    size_t A_size = size_t(lda) * N;
    size_t X_size = size_t(incx) * N;
    size_t Y_size = size_t(incy) * N;

    int batch_count = argus.batch_count;

    hipblasFillMode_t uplo = char2hipblas_fill(argus.uplo_option);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || lda < N || incx <= 0 || incy <= 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasLocalHandle handle(argus);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    // arrays of pointers-to-host on host
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hx(X_size, 1, batch_count);
    host_batch_vector<T> hy(Y_size, 1, batch_count);
    host_batch_vector<T> hy_host(Y_size, 1, batch_count);
    host_batch_vector<T> hy_device(Y_size, 1, batch_count);
    host_batch_vector<T> hy_cpu(Y_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(X_size, 1, batch_count);
    device_batch_vector<T> dy(Y_size, 1, batch_count);
    device_vector<T>       d_alpha(1);
    device_vector<T>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    // Initial Data on CPU
    hipblas_init(hA, true);
    hipblas_init(hx);
    hipblas_init(hy);
    hy_cpu.copy_from(hy);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasHemvBatchedFn(handle,
                                             uplo,
                                             N,
                                             (T*)&h_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             dx.ptr_on_device(),
                                             incx,
                                             (T*)&h_beta,
                                             dy.ptr_on_device(),
                                             incy,
                                             batch_count));

    CHECK_HIP_ERROR(hy_host.transfer_from(dy));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasHemvBatchedFn(handle,
                                             uplo,
                                             N,
                                             d_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             dx.ptr_on_device(),
                                             incx,
                                             d_beta,
                                             dy.ptr_on_device(),
                                             incy,
                                             batch_count));

    CHECK_HIP_ERROR(hy_device.transfer_from(dy));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_hemv<T>(uplo, N, h_alpha, hA[b], lda, hx[b], incx, h_beta, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incy, hy_cpu, hy_host);
            unit_check_general<T>(1, N, batch_count, incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, N, incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', 1, N, incy, hy_cpu, hy_device, batch_count);
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(dy.transfer_from(hy));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasHemvBatchedFn(handle,
                                                     uplo,
                                                     N,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     d_beta,
                                                     dy.ptr_on_device(),
                                                     incy,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_alpha, e_lda, e_incx, e_beta, e_incy, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            hemv_gflop_count<T>(N),
            hemv_gbyte_count<T>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
