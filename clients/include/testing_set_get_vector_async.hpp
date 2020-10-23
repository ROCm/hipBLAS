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
hipblasStatus_t testing_set_get_vector_async(const Arguments& argus)
{
    bool FORTRAN                 = argus.fortran;
    auto hipblasSetVectorAsyncFn = FORTRAN ? hipblasSetVectorAsyncFortran : hipblasSetVectorAsync;
    auto hipblasGetVectorAsyncFn = FORTRAN ? hipblasGetVectorAsyncFortran : hipblasGetVectorAsync;

    int M    = argus.M;
    int incx = argus.incx;
    int incy = argus.incy;
    int incd = argus.incd;

    hipblasStatus_t status     = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_set = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_get = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || incx <= 0 || incy <= 0 || incd <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> hx(M * incx);
    host_vector<T> hy(M * incy);
    host_vector<T> hy_ref(M * incy);

    device_vector<T> db(M * incd);

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    hipStream_t stream;
    hipblasGetStream(handle, &stream);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, M, incy);
    hy_ref = hy;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status_set
        = hipblasSetVectorAsyncFn(M, sizeof(T), (void*)hx.data(), incx, (void*)db, incd, stream);
    status_get
        = hipblasGetVectorAsyncFn(M, sizeof(T), (void*)db, incd, (void*)hy.data(), incy, stream);

    hipStreamSynchronize(stream);

    if(status_set != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status_set;
    }

    if(status_get != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status_get;
    }

    if(argus.unit_check)
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
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
