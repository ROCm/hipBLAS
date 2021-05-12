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
hipblasStatus_t testing_symv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSymvStridedBatchedFn
        = FORTRAN ? hipblasSymvStridedBatched<T, true> : hipblasSymvStridedBatched<T, false>;

    int    M            = argus.M;
    int    lda          = argus.lda;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasStride stride_A = lda * M * stride_scale;
    hipblasStride stride_x = M * incx * stride_scale;
    hipblasStride stride_y = M * incy * stride_scale;

    size_t A_size = stride_A * batch_count;
    size_t X_size = stride_x * batch_count;
    size_t Y_size = stride_y * batch_count;

    hipblasFillMode_t uplo = char2hipblas_fill(argus.uplo_option);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || lda < M || incx == 0 || incy == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hy_cpu(Y_size);
    host_vector<T> hy_host(Y_size);
    host_vector<T> hy_device(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    T h_alpha = argus.get_alpha<T>();
    T h_beta  = argus.get_beta<T>();

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, M, lda, stride_A, batch_count);
    hipblas_init<T>(hx, 1, M, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, M, incy, stride_y, batch_count);
    hy_cpu = hy;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasSymvStridedBatchedFn(handle,
                                                    uplo,
                                                    M,
                                                    &h_alpha,
                                                    dA,
                                                    lda,
                                                    stride_A,
                                                    dx,
                                                    incx,
                                                    stride_x,
                                                    &h_beta,
                                                    dy,
                                                    incy,
                                                    stride_y,
                                                    batch_count));

    CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyDeviceToHost));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasSymvStridedBatchedFn(handle,
                                                    uplo,
                                                    M,
                                                    d_alpha,
                                                    dA,
                                                    lda,
                                                    stride_A,
                                                    dx,
                                                    incx,
                                                    stride_x,
                                                    d_beta,
                                                    dy,
                                                    incy,
                                                    stride_y,
                                                    batch_count));

    CHECK_HIP_ERROR(hipMemcpy(hy_device.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_symv<T>(uplo,
                          M,
                          h_alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hx.data() + b * stride_x,
                          incx,
                          h_beta,
                          hy_cpu.data() + b * stride_y,
                          incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, incy, stride_y, hy_cpu, hy_host);
            unit_check_general<T>(1, M, batch_count, incy, stride_y, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', 1, M, incy, stride_y, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', 1, M, incy, stride_y, hy_cpu, hy_device, batch_count);
        }
    }

    if(argus.timing)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSymvStridedBatchedFn(handle,
                                                            uplo,
                                                            M,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_A,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            d_beta,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option,
                      e_M,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         symv_gflop_count<T>(M),
                         symv_gbyte_count<T>(M),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
