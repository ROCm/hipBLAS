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
hipblasStatus_t testing_set_get_vector_async(const Arguments& argus)
{
    bool FORTRAN                 = argus.fortran;
    auto hipblasSetVectorAsyncFn = FORTRAN ? hipblasSetVectorAsyncFortran : hipblasSetVectorAsync;
    auto hipblasGetVectorAsyncFn = FORTRAN ? hipblasGetVectorAsyncFortran : hipblasGetVectorAsync;

    int M    = argus.M;
    int incx = argus.incx;
    int incy = argus.incy;
    int incd = argus.incd;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incd <= 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * incx);
    host_vector<T> hy(M * incy);
    host_vector<T> hy_ref(M * incy);

    device_vector<T> db(M * incd);

    double             hipblas_error = 0.0, gpu_time_used = 0.0;
    hipblasLocalHandle handle(argus);

    hipStream_t stream;
    hipblasGetStream(handle, &stream);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, M, incy);
    hy_ref = hy;

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(
        hipblasSetVectorAsyncFn(M, sizeof(T), (void*)hx, incx, (void*)db, incd, stream));
    CHECK_HIPBLAS_ERROR(
        hipblasGetVectorAsyncFn(M, sizeof(T), (void*)db, incd, (void*)hy, incy, stream));

    hipStreamSynchronize(stream);

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int i = 0; i < M; i++)
        {
            hy_ref[i * incy] = hx[i * incx];
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, incy, hy.data(), hy_ref.data());
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, incy, hy, hy_ref);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasSetVectorAsyncFn(M, sizeof(T), (void*)hx, incx, (void*)db, incd, stream));
            CHECK_HIPBLAS_ERROR(
                hipblasGetVectorAsyncFn(M, sizeof(T), (void*)db, incd, (void*)hy, incy, stream));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_incx, e_incy, e_incb>{}.log_args<T>(std::cout,
                                                                 argus,
                                                                 gpu_time_used,
                                                                 ArgumentLogging::NA_value,
                                                                 set_get_vector_gbyte_count<T>(M),
                                                                 hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
