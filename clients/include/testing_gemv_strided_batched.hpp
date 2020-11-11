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
hipblasStatus_t testing_gemvStridedBatched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemvStridedBatchedFn
        = FORTRAN ? hipblasGemvStridedBatched<T, true> : hipblasGemvStridedBatched<T, false>;

    int    M            = argus.M;
    int    N            = argus.N;
    int    lda          = argus.lda;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int stride_A = lda * N * stride_scale;
    int stride_x;
    int stride_y;

    int A_size = stride_A * batch_count;
    int X_size;
    int Y_size;

    int x_els;
    int y_els;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    // transA = HIPBLAS_OP_T;
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
    if(M < 0 || N < 0 || lda < 0 || incx <= 0 || incy <= 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size);
    host_vector<T> hy(Y_size);
    host_vector<T> hz(Y_size);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size);
    device_vector<T> dy(Y_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda, stride_A, batch_count);
    hipblas_init<T>(hx, 1, x_els, incx, stride_x, batch_count);
    hipblas_init<T>(hy, 1, y_els, incy, stride_y, batch_count);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * X_size, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * Y_size, hipMemcpyHostToDevice);

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */

        status = hipblasGemvStridedBatchedFn(handle,
                                             transA,
                                             M,
                                             N,
                                             (T*)&alpha,
                                             dA,
                                             lda,
                                             stride_A,
                                             dx,
                                             incx,
                                             stride_x,
                                             (T*)&beta,
                                             dy,
                                             incy,
                                             stride_y,
                                             batch_count);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            // here in cuda
            hipblasDestroy(handle);
            return status;
        }

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_gemv<T>(transA,
                          M,
                          N,
                          alpha,
                          hA.data() + b * stride_A,
                          lda,
                          hx.data() + b * stride_x,
                          incx,
                          beta,
                          hz.data() + b * stride_y,
                          incy);
        }

        // copy output from device to CPU
        hipMemcpy(hy.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, y_els, batch_count, incy, stride_y, hz, hy);
        }
        if(argus.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, y_els, incy, stride_y, hz, hy, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        status = hipblasGetStream(handle, &stream);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
            {
                gpu_time_used = get_time_us_sync(stream);
            }
            status = hipblasGemvStridedBatchedFn(handle,
                                                 transA,
                                                 M,
                                                 N,
                                                 (T*)&alpha,
                                                 dA,
                                                 lda,
                                                 stride_A,
                                                 dx,
                                                 incx,
                                                 stride_x,
                                                 (T*)&beta,
                                                 dy,
                                                 incy,
                                                 stride_y,
                                                 batch_count);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                // here in cuda
                hipblasDestroy(handle);
                return status;
            }
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA_option,
                      e_M,
                      e_N,
                      e_stride_a,
                      e_alpha,
                      e_lda,
                      e_incx,
                      e_stride_x,
                      e_beta,
                      e_incy,
                      e_stride_y,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         gemv_gflop_count<T>(transA, M, N),
                         gemv_gbyte_count<T>(transA, M, N),
                         rocblas_error);
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
