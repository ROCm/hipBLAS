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
hipblasStatus_t testing_gemvBatched(Arguments argus)
{

    int M    = argus.M;
    int N    = argus.N;
    int lda  = argus.lda;
    int incx = argus.incx;
    int incy = argus.incy;

    int A_size = lda * N;
    int X_size;
    int Y_size;

    int batch_count = argus.batch_count;

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
    if(M < 0 || N < 0 || lda < 0 || incx <= 0 || incy <= 0 || batch_count < 0) // TODO: batch_count <= 0?
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    hipblasHandle_t handle;
    hipblasCreate(&handle);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    T alpha = (T)argus.alpha;
    T beta  = (T)argus.beta;

    // malloc arrays of pointers-to-host on host
    T** hA_array = (T**)malloc(batch_count * sizeof(T*));
    T** hx_array = (T**)malloc(batch_count * sizeof(T*));
    T** hy_array = (T**)malloc(batch_count * sizeof(T*));
    T** hz_array = (T**)malloc(batch_count * sizeof(T*));
    // vector<T> hA(A_size);
    // vector<T> hx(X_size * incx);
    // vector<T> hy(Y_size * incy);
    // vector<T> hz(Y_size * incy);

    // malloc arrays of pointers-to-device on host (device_batch_pointer in rocBlas)
    T** bA_array = (T**)malloc(batch_count * sizeof(T*));
    T** bx_array = (T**)malloc(batch_count * sizeof(T*));
    T** by_array = (T**)malloc(batch_count * sizeof(T*));
    // T** bz_array = (T**)malloc(batch_count * sizeof(T*));

    // Arrays of pointers-to-device on device (device_pointer in rocBlas)
    T** dA_array = nullptr;
    T** dx_array = nullptr;
    T** dy_array = nullptr;
    // T** dz_array = nullptr;

    if((!hA_array) || (!hx_array) || (!hy_array) || (!hz_array) || (!bA_array) || (!bx_array) || (!by_array))
    {
        // hipFree(stuff); // TODO
        hipblasDestroy(handle);
        std::cerr << "malloc error\n";
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // malloc arrays of pointers-to-device on device
    hipError_t err_A, err_x, err_y;
    err_A = hipMalloc((void**)&dA_array, batch_count * sizeof(T*));
    err_x = hipMalloc((void**)&dx_array, batch_count * sizeof(T*));
    err_y = hipMalloc((void**)&dy_array, batch_count * sizeof(T*));

    if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
    {
        // hipFree(stuff); // TODO
        hipblasDestroy(handle);
        std::cerr << "hipMalloc error\n";
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        // malloc matrices on host
        hA_array[b] = (T*)malloc(A_size * sizeof(T));
        hx_array[b] = (T*)malloc(X_size * sizeof(T));
        hy_array[b] = (T*)malloc(Y_size * sizeof(T));
    
        if(!hA_array[b] || !hx_array[b] || !hy_array[b])
        {
            // hipFree(stuff); // TODO
            hipblasDestroy(handle);
            std::cerr << "hX_array[i] malloc error\n";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

        // malloc matrices on device
        err_A = hipMalloc((void**)&dA_array[b], A_size * sizeof(T));
        err_x = hipMalloc((void**)&dx_array[b], X_size * sizeof(T));
        err_y = hipMalloc((void**)&dy_array[b], Y_size * sizeof(T));

        

        hipblas_init<T>(hA, M, N, lda);
        hipblas_init<T>(hx, 1, X_size, incx);
        hipblas_init<T>(hy, 1, Y_size, incy);
    }

    // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
    hz = hy;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(T) * lda * N, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(T) * X_size * incx, hipMemcpyHostToDevice);
    hipMemcpy(dy, hy.data(), sizeof(T) * Y_size * incy, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {

        status = hipblasGemv<T>(
            handle, transA, M, N, (T*)&alpha, dA, lda, dx, incx, (T*)&beta, dy, incy);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            CHECK_HIP_ERROR(hipFree(dA));
            CHECK_HIP_ERROR(hipFree(dx));
            CHECK_HIP_ERROR(hipFree(dy));
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hy.data(), dy, sizeof(T) * Y_size * incy, hipMemcpyDeviceToHost);

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        cblas_gemv<T>(transA, M, N, alpha, hA.data(), lda, hx.data(), incx, beta, hz.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, incy, hz.data(), hy.data());
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
