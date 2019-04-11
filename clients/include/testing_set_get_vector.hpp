/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_set_get_vector(Arguments argus)
{
    int M    = argus.M;
    int incx = argus.incx;
    int incy = argus.incy;
    int incd = argus.incd;

    hipblasStatus_t status     = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_set = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_get = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0)
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
    else if(incd <= 0)
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    vector<T> hx(M * incx);
    vector<T> hy(M * incy);
    vector<T> hy_ref(M * incy);

    T* db;

    hipblasHandle_t handle;

    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&db, M * incd * sizeof(T)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, M, incx);
    hipblas_init<T>(hy, 1, M, incy);
    hy_ref = hy;

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    status_set = hipblasSetVector(M, sizeof(T), (void*)hx.data(), incx, (void*)db, incd);

    status_get = hipblasGetVector(M, sizeof(T), (void*)db, incd, (void*)hy.data(), incy);

    if(status_set != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(db));
        hipblasDestroy(handle);
        return status;
    }

    if(status_get != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(db));
        hipblasDestroy(handle);
        return status;
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

    CHECK_HIP_ERROR(hipFree(db));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
