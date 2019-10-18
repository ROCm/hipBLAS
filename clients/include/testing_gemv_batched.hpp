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

#define CLEANUP()                            \
    do                                       \
    {                                        \
        for(int b = 0; b < batch_count; b++) \
        {                                    \
            if(bA_array[b])                  \
                hipFree(bA_array[b]);        \
            if(bx_array[b])                  \
                hipFree(bx_array[b]);        \
            if(by_array[b])                  \
                hipFree(by_array[b]);        \
                                             \
            if(hA_array[b])                  \
                free(hA_array[b]);           \
            if(hx_array[b])                  \
                free(hx_array[b]);           \
            if(hy_array[b])                  \
                free(hy_array[b]);           \
        }                                    \
                                             \
        if(hA_array)                         \
            free(hA_array);                  \
        if(hx_array)                         \
            free(hx_array);                  \
        if(hy_array)                         \
            free(hy_array);                  \
        if(hz_array)                         \
            free(hz_array);                  \
                                             \
        if(bA_array)                         \
            free(bA_array);                  \
        if(bx_array)                         \
            free(bx_array);                  \
        if(by_array)                         \
            free(by_array);                  \
                                             \
        if(dA_array)                         \
            hipFree(dA_array);               \
        if(dx_array)                         \
            hipFree(dx_array);               \
        if(dy_array)                         \
            hipFree(dy_array);               \
    } while(0)

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
    if(M < 0 || N < 0 || lda < 0 || incx <= 0 || incy <= 0
       || batch_count < 0) // TODO: batch_count <= 0?
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

    // malloc arrays of pointers-to-device on host (device_batch_pointer in rocBlas)
    T** bA_array = (T**)malloc(batch_count * sizeof(T*));
    T** bx_array = (T**)malloc(batch_count * sizeof(T*));
    T** by_array = (T**)malloc(batch_count * sizeof(T*));
    // T** bz_array = (T**)malloc(batch_count * sizeof(T*));

    // Arrays of pointers-to-device on device (device_pointer in rocBlas)
    T** dA_array = nullptr;
    T** dx_array = nullptr;
    T** dy_array = nullptr;

    if((!hA_array) || (!hx_array) || (!hy_array) || (!hz_array) || (!bA_array) || (!bx_array)
       || (!by_array))
    {
        CLEANUP();
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
        CLEANUP();
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
        hz_array[b] = (T*)malloc(Y_size * sizeof(T));

        if(!hA_array[b] || !hx_array[b] || !hy_array[b] || !hz_array[b])
        {
            CLEANUP();
            hipblasDestroy(handle);
            std::cerr << "hX_array[i] malloc error\n";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

        // malloc matrices on device
        err_A = hipMalloc((void**)&bA_array[b], A_size * sizeof(T));
        err_x = hipMalloc((void**)&bx_array[b], X_size * sizeof(T));
        err_y = hipMalloc((void**)&by_array[b], Y_size * sizeof(T));

        if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
        {
            CLEANUP();
            hipblasDestroy(handle);
            std::cerr << "bx_array[i] hipMalloc error\n";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }

        // initialize matrices on host
        srand(1);
        hipblas_init<T>(hA_array[b], M, N, lda);
        hipblas_init<T>(hx_array[b], 1, X_els, incx);
        hipblas_init<T>(hy_array[b], 1, Y_els, incy);

        err_A = hipMemcpy(bA_array[b], hA_array[b], sizeof(T) * A_size, hipMemcpyHostToDevice);
        err_x = hipMemcpy(bx_array[b], hx_array[b], sizeof(T) * X_size, hipMemcpyHostToDevice);
        err_y = hipMemcpy(by_array[b], hy_array[b], sizeof(T) * Y_size, hipMemcpyHostToDevice);

        // copy vector is easy in STL; hz = hy: save a copy in hz which will be output of CPU BLAS
        hz_array[b] = hy_array[b];

        if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
        {
            CLEANUP();
            hipblasDestroy(handle);
            std::cerr << "bX_array[i] hipMemcpy error\n";
            return HIPBLAS_STATUS_MAPPING_ERROR;
        }
    }

    err_A = hipMemcpy(dA_array, bA_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_x = hipMemcpy(dx_array, bx_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    err_y = hipMemcpy(dy_array, by_array, batch_count * sizeof(T*), hipMemcpyHostToDevice);
    if(err_A != hipSuccess || err_x != hipSuccess || err_y != hipSuccess)
    {
        CLEANUP();
        hipblasDestroy(handle);
        std::cerr << "dX_array[i] hipMemcpy error\n";
        return HIPBLAS_STATUS_MAPPING_ERROR;
    }

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasGemvBatched<T>(handle,
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
            CLEANUP();
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    for(int b = 0; b < batch_count; b++)
    {
        hipMemcpy(hy_array[b], by_array[b], sizeof(T) * Y_size, hipMemcpyDeviceToHost);
    }

    if(argus.unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_gemv<T>(
                transA, M, N, alpha, hA_array[b], lda, hx_array[b], incx, beta, hz_array[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, Y_size, batch_count, incy, hz_array, hy_array);
        }
    }

    CLEANUP();
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
