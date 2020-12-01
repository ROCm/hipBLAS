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

template <typename Tex, typename Tx = Tex, typename Tcs = Tx>
hipblasStatus_t testing_rot_strided_batched_ex_template(Arguments arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotStridedBatchedExFn
        = FORTRAN ? hipblasRotStridedBatchedExFortran : hipblasRotStridedBatchedEx;

    int    N            = arg.N;
    int    incx         = arg.incx;
    int    incy         = arg.incy;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    int stridex = N * incx * stride_scale;
    int stridey = N * incy * stride_scale;

    int size_x = stridex * batch_count;
    int size_y = stridey * batch_count;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || !batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    // const Tcs rel_error = std::numeric_limits<Tcs>::epsilon() * 1000;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    device_vector<Tx>  dx(size_x);
    device_vector<Tx>  dy(size_y);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);

    // Initial Data on CPU
    host_vector<Tx>  hx(size_x);
    host_vector<Tx>  hy(size_y);
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);
    srand(1);
    hipblas_init<Tx>(hx, 1, N, incx, stridex, batch_count);
    hipblas_init<Tx>(hy, 1, N, incy, stridey, batch_count);

    hipblas_init<Tcs>(hc, 1, 1, 1);
    hipblas_init<Tcs>(hs, 1, 1, 1);

    // CPU BLAS reference data
    host_vector<Tx> hx_cpu = hx;
    host_vector<Tx> hy_cpu = hy;

    if(arg.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rot<Tx, Tcs, Tcs>(
                N, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy, *hc, *hs);
        }

        // Test host
        {
            status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Tx) * size_y, hipMemcpyHostToDevice));
            status_2 = (hipblasRotStridedBatchedExFn(handle,
                                                     N,
                                                     dx,
                                                     xType,
                                                     incx,
                                                     stridex,
                                                     dy,
                                                     yType,
                                                     incy,
                                                     stridey,
                                                     hc,
                                                     hs,
                                                     csType,
                                                     batch_count,
                                                     executionType));

            if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_1 != HIPBLAS_STATUS_SUCCESS)
                    return status_1;
                if(status_2 != HIPBLAS_STATUS_SUCCESS)
                    return status_2;
            }

            host_vector<Tx> rx(size_x);
            host_vector<Tx> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(Tx) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general(1, N, batch_count, incx, stridex, hx_cpu.data(), rx.data());
                unit_check_general(1, N, batch_count, incy, stridey, hy_cpu.data(), ry.data());
            }
        }

        // Test device
        {
            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
            CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, hy, sizeof(Tx) * size_y, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

            if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_3 != HIPBLAS_STATUS_SUCCESS)
                    return status_3;
                if(status_4 != HIPBLAS_STATUS_SUCCESS)
                    return status_4;
            }

            status_3 = (hipblasRotStridedBatchedExFn(handle,
                                                     N,
                                                     dx,
                                                     xType,
                                                     incx,
                                                     stridex,
                                                     dy,
                                                     yType,
                                                     incy,
                                                     stridey,
                                                     dc,
                                                     ds,
                                                     csType,
                                                     batch_count,
                                                     executionType));
            host_vector<Tx> rx(size_x);
            host_vector<Tx> ry(size_y);
            CHECK_HIP_ERROR(hipMemcpy(rx, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(ry, dy, sizeof(Tx) * size_y, hipMemcpyDeviceToHost));
            if(arg.unit_check)
            {
                unit_check_general(1, N, batch_count, incx, stridex, hx_cpu.data(), rx.data());
                unit_check_general(1, N, batch_count, incy, stridey, hy_cpu.data(), ry.data());
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_rot_strided_batched_ex(Arguments arg)
{
    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16B
       && csType == HIPBLAS_R_16B)
    {
        status
            = testing_rot_strided_batched_ex_template<float, hipblasBfloat16, hipblasBfloat16>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16F
            && csType == HIPBLAS_R_16F)
    {
        status = testing_rot_strided_batched_ex_template<float, hipblasHalf, hipblasHalf>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_strided_batched_ex_template<float>(arg);
    }
    else if(executionType == HIPBLAS_R_64F && xType == yType && xType == HIPBLAS_R_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_strided_batched_ex_template<double>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_R_32F)
    {
        status
            = testing_rot_strided_batched_ex_template<hipblasComplex, hipblasComplex, float>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_C_32F)
    {
        status = testing_rot_strided_batched_ex_template<hipblasComplex>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_strided_batched_ex_template<hipblasDoubleComplex,
                                                         hipblasDoubleComplex,
                                                         double>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_C_64F)
    {
        status = testing_rot_strided_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
