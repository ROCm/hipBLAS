/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

// #include "cblas_interface.h"
// #include "hipblas.hpp"
// #include "norm.h"
// #include "unit.h"
// #include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename XY_TYPE, typename E_TYPE = XY_TYPE>
hipblasStatus_t testing_axpy_ex_template(Arguments argus)
{
    bool            FORTRAN         = argus.fortran;
    auto            hipblasAxpyExFn = FORTRAN ? hipblasAxpyExFortran : hipblasAxpyEx;
    hipblasStatus_t status          = HIPBLAS_STATUS_SUCCESS;

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || !incy)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;
    hipblasDatatype_t alphaType     = xType;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    int     sizeX = N * abs_incx;
    int     sizeY = N * abs_incy;
    XY_TYPE alpha = argus.alpha;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<XY_TYPE> hx(sizeX);
    host_vector<XY_TYPE> hy(sizeY);
    host_vector<XY_TYPE> hx_cpu(sizeX);
    host_vector<XY_TYPE> hy_cpu(sizeY);

    device_vector<XY_TYPE> dx(sizeX);
    device_vector<XY_TYPE> dy(sizeX);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<XY_TYPE>(hx, 1, N, abs_incx);
    hipblas_init<XY_TYPE>(hy, 1, N, abs_incy);

    // copy vector is easy in STL; hx_cpu = hx: save a copy in hx_cpu which will be output of CPU BLAS
    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(XY_TYPE) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(XY_TYPE) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasAxpyExFn(
        handle, N, &alpha, alphaType, dx, xType, incx, dy, yType, incy, executionType);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(XY_TYPE) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(XY_TYPE) * sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        cblas_axpy<XY_TYPE>(N, alpha, hx_cpu.data(), incx, hy_cpu.data(), incy);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<XY_TYPE>(1, N, abs_incx, hx_cpu.data(), hx.data());
            unit_check_general<XY_TYPE>(1, N, abs_incy, hy_cpu.data(), hy.data());
        }

    } // end of if unit check

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_axpy_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_ex_template<hipblasHalf, float>(argus);
    }
    else if(xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_ex_template<float>(argus);
    }
    else if(xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_axpy_ex_template<double>(argus);
    }
    else if(xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F && executionType == HIPBLAS_C_32F)
    {
        status = testing_axpy_ex_template<hipblasComplex>(argus);
    }
    else if(xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F && executionType == HIPBLAS_C_64F)
    {
        status = testing_axpy_ex_template<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
