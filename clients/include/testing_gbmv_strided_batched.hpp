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
hipblasStatus_t testing_gbmv_strided_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGbmvStridedBatchedFn
        = FORTRAN ? hipblasGbmvStridedBatched<T, true> : hipblasGbmvStridedBatched<T, false>;

    int    M            = argus.M;
    int    N            = argus.N;
    int    KL           = argus.KL;
    int    KU           = argus.KU;
    int    lda          = argus.lda;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasStride stride_A = lda * N * stride_scale;
    hipblasStride stride_x;
    hipblasStride stride_y;

    int A_size = stride_A * batch_count;
    int X_size;
    int Y_size;

    int x_els;
    int y_els;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    if(transA == HIPBLAS_OP_N)
    {
        x_els = N;
        y_els = M;
    }
    else
    {
        x_els = M;
        y_els = N;
    }

    stride_x = x_els * incx * stride_scale;
    stride_y = y_els * incy * stride_scale;
    X_size   = stride_x * batch_count;
    Y_size   = stride_y * batch_count;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || KL < 0 || KU < 0 || lda < KL + KU + 1 || incx == 0 || incy == 0
       || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hy_host(Y_size);
    host_vector<T> hy_device(Y_size);
    host_vector<T> hy_cpu(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);
    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = (T)argus.alpha;
    T h_beta  = (T)argus.beta;

    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda, stride_A, batch_count);
    hipblas_init<T>(hx, 1, x_els, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, y_els, incy, stride_y, batch_count);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hy_cpu = hy;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice);

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasGbmvStridedBatchedFn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
                                                    (T*)&h_alpha,
                                                    dA,
                                                    lda,
                                                    stride_A,
                                                    dx,
                                                    incx,
                                                    stride_x,
                                                    (T*)&h_beta,
                                                    dy,
                                                    incy,
                                                    stride_y,
                                                    batch_count));

    CHECK_HIP_ERROR(hipMemcpy(hy_host.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasGbmvStridedBatchedFn(handle,
                                                    transA,
                                                    M,
                                                    N,
                                                    KL,
                                                    KU,
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

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_gbmv<T>(transA,
                          M,
                          N,
                          KL,
                          KU,
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
            unit_check_general<T>(1, y_els, batch_count, incy, stride_y, hy_cpu, hy_host);
            unit_check_general<T>(1, y_els, batch_count, incy, stride_y, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host = norm_check_general<T>(
                'F', 1, y_els, incy, stride_y, hy_cpu, hy_host, batch_count);
            hipblas_error_device = norm_check_general<T>(
                'F', 1, y_els, incy, stride_y, hy_cpu, hy_device, batch_count);
        }
    }

    if(argus.timing)
    {
        hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice);
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGbmvStridedBatchedFn(handle,
                                                            transA,
                                                            M,
                                                            N,
                                                            KL,
                                                            KU,
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

        ArgumentModel<e_M,
                      e_N,
                      e_KL,
                      e_KU,
                      e_lda,
                      e_stride_a,
                      e_incx,
                      e_stride_x,
                      e_incy,
                      e_stride_y>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         gbmv_gflop_count<T>(transA, M, N, KL, KU),
                         gbmv_gbyte_count<T>(transA, M, N, KL, KU),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
