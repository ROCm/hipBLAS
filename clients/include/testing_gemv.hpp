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
hipblasStatus_t testing_gemv(const Arguments& argus)
{
    bool FORTRAN       = argus.fortran;
    auto hipblasGemvFn = FORTRAN ? hipblasGemv<T, true> : hipblasGemv<T, false>;

    int M    = argus.M;
    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;

    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    // transA = HIPBLAS_OP_T;
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

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    if(N < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(lda < 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incx <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }
    else if(incy <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hA(A_size);
    host_vector<T> hx(X_size * incx);
    host_vector<T> hy(Y_size * incy);
    host_vector<T> hz(Y_size * incy);

    device_vector<T> dA(A_size);
    device_vector<T> dx(X_size * incx);
    device_vector<T> dy(Y_size * incy);

    double gpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hA, M, N, lda);
    hipblas_init<T>(hx, 1, X_size, incx);
    hipblas_init<T>(hy, 1, Y_size, incy);

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * X_size * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * Y_size * incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */

    if(argus.unit_check || argus.norm_check)
    {
        status = hipblasGemvFn(
            handle, transA, M, N, (T*)&alpha, dA, lda, dx, incx, (T*)&beta, dy, incy);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }

        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_gemv<T>(transA, M, N, alpha, hA.data(), lda, hx.data(), incx, beta, hz.data(), incy);

        // copy output from device to CPU
        hipMemcpy(hy.data(), dy, sizeof(T) * Y_size * incy, hipMemcpyDeviceToHost);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, incy, hz, hy);
        }
        if(argus.norm_check)
        {
            rocblas_error = norm_check_general<T>('F', 1, Y_size, incy, hz, hy);
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

            status = hipblasGemvFn(
                handle, transA, M, N, (T*)&alpha, dA, lda, dx, incx, (T*)&beta, dy, incy);

            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                hipblasDestroy(handle);
                return status;
            }
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA_option, e_M, e_N, e_alpha, e_lda, e_incx, e_beta, e_incy>{}
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
