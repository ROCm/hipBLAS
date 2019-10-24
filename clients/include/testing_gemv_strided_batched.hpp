/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_gemvStridedBatched(Arguments argus)
{
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

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasGemvStridedBatched<T>(handle,
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

    // copy output from device to CPU
    hipMemcpy(hy.data(), dy, sizeof(T) * Y_size, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
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

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, y_els, batch_count, incy, stride_y, hz, hy);
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
