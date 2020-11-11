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
hipblasStatus_t testing_gemvBatched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGemvBatchedFn
        = FORTRAN ? hipblasGemvBatched<T, true> : hipblasGemvBatched<T, false>;

    int M    = argus.M;
    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;
    int X_els;
    int Y_els;

    int batch_count = argus.batch_count;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    // transA = HIPBLAS_OP_T;
    if(transA == HIPBLAS_OP_N)
    {
        X_els = N;
        Y_els = M;
    }
    else
    {
        X_els = M;
        Y_els = N;
    }
    X_size = X_els * incx;
    Y_size = Y_els * incy;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < 0 || incx <= 0 || incy <= 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    // arrays of pointers-to-host on host
    host_vector<T> hA_array[batch_count];
    host_vector<T> hx_array[batch_count];
    host_vector<T> hy_array[batch_count];
    host_vector<T> hz_array[batch_count];

    // arrays of pointers-to-device on host
    device_batch_vector<T> bA_array(batch_count, A_size);
    device_batch_vector<T> bx_array(batch_count, X_size);
    device_batch_vector<T> by_array(batch_count, Y_size);

    // arrays of pointers-to-device on device
    device_vector<T*, 0, T> dA_array(batch_count);
    device_vector<T*, 0, T> dx_array(batch_count);
    device_vector<T*, 0, T> dy_array(batch_count);

    int last = batch_count - 1;
    if(!dA_array || !dx_array || !dy_array || (!bA_array[last] && A_size)
       || (!bx_array[last] && X_size) || (!by_array[last] && Y_size))
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    hipError_t err_A, err_x, err_y;
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hA_array[b] = host_vector<T>(A_size);
        hx_array[b] = host_vector<T>(X_size);
        hy_array[b] = host_vector<T>(Y_size);
        hz_array[b] = host_vector<T>(Y_size);

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[b], M, N, lda);
        hipblas_init<T>(hx_array[b], 1, X_els, incx);
        hipblas_init<T>(hy_array[b], 1, Y_els, incy);

        hz_array[b] = hy_array[b];
        err_A = hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice);
        err_x = hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * X_size, hipMemcpyHostToDevice);
        err_y = hipMemcpy(by_array[b], hy_array[b], sizeof(T) * Y_size, hipMemcpyHostToDevice);

        if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
        {
            hipblasDestroy(handle);
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    err_A = hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_x = hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_y = hipMemcpy(dy_array, by_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
    {
        hipblasDestroy(handle);
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        status = hipblasGemvBatchedFn(handle,
                                      transA,
                                      M,
                                      N,
                                      (T*)&alpha,
                                      dA_array,
                                      lda,
                                      dx_array,
                                      incx,
                                      (T*)&beta,
                                      dy_array,
                                      incy,
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
            cblas_gemv<T>(
                transA, M, N, alpha, hA_array[b], lda, hx_array[b], incx, beta, hz_array[b], incy);
        }

        // copy output from device to CPU
        for(int b = 0; b < batch_count; b++)
        {
            hipMemcpy(hy_array[b], by_array[b], sizeof(T) * Y_size, hipMemcpyDeviceToHost);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, batch_count, incy, hz_array, hy_array);
        }
        if(argus.norm_check)
        {
            rocblas_error
                = norm_check_general<T>('F', 1, Y_size, incy, hz_array, hy_array, batch_count);
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
            status = hipblasGemvBatchedFn(handle,
                                          transA,
                                          M,
                                          N,
                                          (T*)&alpha,
                                          dA_array,
                                          lda,
                                          dx_array,
                                          incx,
                                          (T*)&beta,
                                          dy_array,
                                          incy,
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
                      e_alpha,
                      e_lda,
                      e_incx,
                      e_beta,
                      e_incy,
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
