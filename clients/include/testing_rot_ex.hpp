/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "hipblas.hpp"
#include "near.h"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename E_TYPE, typename X_TYPE = E_TYPE, typename CS_TYPE = X_TYPE>
hipblasStatus_t testing_rot_ex_template(Arguments arg)
{
    bool FORTRAN        = arg.fortran;
    auto hipblasRotExFn = FORTRAN ? hipblasRotExFortran : hipblasRotEx;

    int N    = arg.N;
    int incx = arg.incx;
    int incy = arg.incy;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    const CS_TYPE rel_error = std::numeric_limits<CS_TYPE>::epsilon() * 1000;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    device_vector<X_TYPE>  dx(size_x);
    device_vector<X_TYPE>  dy(size_y);
    device_vector<CS_TYPE> dc(1);
    device_vector<CS_TYPE> ds(1);

    // Initial Data on CPU
    host_vector<X_TYPE>  hx(size_x);
    host_vector<X_TYPE>  hy(size_y);
    host_vector<CS_TYPE> hc(1);
    host_vector<CS_TYPE> hs(1);
    srand(1);
    hipblas_init<X_TYPE>(hx, 1, N, incx);
    hipblas_init<X_TYPE>(hy, 1, N, incy);

    // Random alpha (0 - 10)
    host_vector<int> alpha(1);
    hipblas_init<int>(alpha, 1, 1, 1);

    // cos and sin of alpha (in rads)
    hc[0] = cos(alpha[0]);
    hs[0] = sin(alpha[0]);

    // CPU BLAS reference data
    host_vector<X_TYPE> cx = hx;
    host_vector<X_TYPE> cy = hy;

    cblas_rot<X_TYPE, CS_TYPE, CS_TYPE>(N, cx.data(), incx, cy.data(), incy, *hc, *hs);

    if(arg.unit_check)
    {
        // Test host
        {
            status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(X_TYPE) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(X_TYPE) * size_y, hipMemcpyHostToDevice));
            status_2 = (hipblasRotExFn(
                handle, N, dx, xType, incx, dy, yType, incy, hc, hs, csType, executionType));

            host_vector<X_TYPE> rx(size_x);
            host_vector<X_TYPE> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(X_TYPE) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(X_TYPE) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, incy, cy.data(), ry.data(), double(rel_error));
            }
        }

        // Test device
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(X_TYPE) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(X_TYPE) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(CS_TYPE), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(CS_TYPE), hipMemcpyHostToDevice));
            status_3 = (hipblasRotExFn(
                handle, N, dx, xType, incx, dy, yType, incy, dc, ds, csType, executionType));
            host_vector<X_TYPE> rx(size_x);
            host_vector<X_TYPE> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(X_TYPE) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(X_TYPE) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                near_check_general(1, N, incx, cx.data(), rx.data(), double(rel_error));
                near_check_general(1, N, incy, cy.data(), ry.data(), double(rel_error));
            }
        }
    }
    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
       || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
    {
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
        if(status_3 != HIPBLAS_STATUS_SUCCESS)
            return status_3;
        if(status_4 != HIPBLAS_STATUS_SUCCESS)
            return status_4;
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_rot_ex(Arguments arg)
{
    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16B
       && csType == HIPBLAS_R_16B)
    {
        status = testing_rot_ex_template<float, hipblasBfloat16, hipblasBfloat16>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16F
            && csType == HIPBLAS_R_16F)
    {
        status = testing_rot_ex_template<float, hipblasHalf, hipblasHalf>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_ex_template<float>(arg);
    }
    else if(executionType == HIPBLAS_R_64F && xType == yType && xType == HIPBLAS_R_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_ex_template<double>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_ex_template<hipblasComplex, hipblasComplex, float>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_C_32F)
    {
        status = testing_rot_ex_template<hipblasComplex>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_ex_template<hipblasDoubleComplex, hipblasDoubleComplex, double>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_C_64F)
    {
        status = testing_rot_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
