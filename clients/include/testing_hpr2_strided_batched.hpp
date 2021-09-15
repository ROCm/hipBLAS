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
hipblasStatus_t testing_hpr2_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasHpr2StridedBatchedFn
        = FORTRAN ? hipblasHpr2StridedBatched<T, true> : hipblasHpr2StridedBatched<T, false>;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int               abs_incx = incx >= 0 ? incx : -incx;
    int               abs_incy = incy >= 0 ? incy : -incy;
    size_t            dim_A    = size_t(N) * (N + 1) / 2;
    hipblasStride     stride_A = dim_A * stride_scale;
    hipblasStride     stride_x = size_t(N) * abs_incx * stride_scale;
    hipblasStride     stride_y = size_t(N) * abs_incy * stride_scale;
    size_t            A_size   = stride_A * batch_count;
    size_t            x_size   = stride_x * batch_count;
    size_t            y_size   = stride_y * batch_count;
    hipblasFillMode_t uplo     = char2hipblas_fill(argus.uplo_option);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = N < 0 || !incx || !incy || batch_count < 0;
    if(invalid_size || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasHpr2StridedBatchedFn(handle,
                                                             uplo,
                                                             N,
                                                             nullptr,
                                                             nullptr,
                                                             incx,
                                                             stride_x,
                                                             nullptr,
                                                             incy,
                                                             stride_y,
                                                             nullptr,
                                                             stride_A,
                                                             batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hA_cpu(A_size);
    host_vector<T> hA_host(A_size);
    host_vector<T> hA_device(A_size);
    host_vector<T> hx(x_size);
    host_vector<T> hy(y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(x_size);
    device_vector<T> dy(y_size);
    device_vector<T> d_alpha(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = argus.get_alpha<T>();

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, 1, dim_A, 1, stride_A, batch_count);
    hipblas_init<T>(hx, 1, N, abs_incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, N, abs_incy, stride_y, batch_count);

    // copy matrix is easy in STL; hA_cpu = hA: save a copy in hA_cpu which will be output of CPU BLAS
    hA_cpu = hA;

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * x_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * y_size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasHpr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        (T*)&h_alpha,
                                                        dx,
                                                        incx,
                                                        stride_x,
                                                        dy,
                                                        incy,
                                                        stride_y,
                                                        dA,
                                                        stride_A,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_host.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasHpr2StridedBatchedFn(handle,
                                                        uplo,
                                                        N,
                                                        d_alpha,
                                                        dx,
                                                        incx,
                                                        stride_x,
                                                        dy,
                                                        incy,
                                                        stride_y,
                                                        dA,
                                                        stride_A,
                                                        batch_count));

        CHECK_HIP_ERROR(hipMemcpy(hA_device.data(), dA, sizeof(T) * A_size, hipMemcpyDeviceToHost));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_hpr2<T>(uplo,
                          N,
                          h_alpha,
                          hx.data() + b * stride_x,
                          incx,
                          hy.data() + b * stride_y,
                          incy,
                          hA_cpu.data() + b * stride_A);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(
                1, dim_A, batch_count, 1, stride_A, hA_cpu.data(), hA_host.data());
            unit_check_general<T>(
                1, dim_A, batch_count, 1, stride_A, hA_cpu.data(), hA_device.data());
        }
        if(argus.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, dim_A, 1, stride_A, hA_cpu.data(), hA_host.data(), batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, dim_A, 1, stride_A, hA_cpu.data(), hA_device.data(), batch_count);
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasHpr2StridedBatchedFn(handle,
                                                            uplo,
                                                            N,
                                                            d_alpha,
                                                            dx,
                                                            incx,
                                                            stride_x,
                                                            dy,
                                                            incy,
                                                            stride_y,
                                                            dA,
                                                            stride_A,
                                                            batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N,
                      e_alpha,
                      e_incx,
                      e_stride_x,
                      e_incy,
                      e_stride_y,
                      e_stride_a,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         hpr2_gflop_count<T>(N),
                         hpr2_gbyte_count<T>(N),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
