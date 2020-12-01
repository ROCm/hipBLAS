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
hipblasStatus_t testing_rot_batched_ex_template(Arguments arg)
{
    bool FORTRAN               = arg.fortran;
    auto hipblasRotBatchedExFn = FORTRAN ? hipblasRotBatchedExFortran : hipblasRotBatchedEx;

    int N           = arg.N;
    int incx        = arg.incx;
    int incy        = arg.incy;
    int batch_count = arg.batch_count;

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || incy <= 0 || batch_count == 0)
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

    size_t size_x = N * size_t(incx);
    size_t size_y = N * size_t(incy);

    host_vector<Tx>  hx[batch_count];
    host_vector<Tx>  hy[batch_count];
    host_vector<Tx>  hx_cpu[batch_count];
    host_vector<Tx>  hy_cpu[batch_count];
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);

    device_batch_vector<Tx> bx(batch_count, size_x);
    device_batch_vector<Tx> by(batch_count, size_y);

    device_vector<Tx*, 0, Tx> dx(batch_count);
    device_vector<Tx*, 0, Tx> dy(batch_count);
    device_vector<Tcs>        dc(1);
    device_vector<Tcs>        ds(1);

    for(int b = 0; b < batch_count; b++)
    {
        hx[b]     = host_vector<Tx>(size_x);
        hy[b]     = host_vector<Tx>(size_y);
        hx_cpu[b] = host_vector<Tx>(size_x);
        hy_cpu[b] = host_vector<Tx>(size_y);

        srand(1);
        hipblas_init<Tx>(hx[b], 1, N, incx);
        hipblas_init<Tx>(hy[b], 1, N, incy);

        hx_cpu[b] = hx[b];
        hy_cpu[b] = hy[b];

        CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(Tx) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(by[b], hy[b], sizeof(Tx) * size_y, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(Tx*) * batch_count, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(Tx*) * batch_count, hipMemcpyHostToDevice));

    hipblas_init<Tcs>(hc, 1, 1, 1);
    hipblas_init<Tcs>(hs, 1, 1, 1);

    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        for(int b = 0; b < batch_count; b++)
        {
            cblas_rot<Tx, Tcs, Tcs>(N, hx_cpu[b].data(), incx, hy_cpu[b].data(), incy, *hc, *hs);
        }

        // Test host
        {
            status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
            status_2 = (hipblasRotBatchedExFn(handle,
                                              N,
                                              dx,
                                              xType,
                                              incx,
                                              dy,
                                              yType,
                                              incy,
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

            host_vector<Tx> rx[batch_count];
            host_vector<Tx> ry[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                rx[b] = host_vector<Tx>(size_x);
                ry[b] = host_vector<Tx>(size_y);
                CHECK_HIP_ERROR(
                    hipMemcpy(rx[b], bx[b], sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(
                    hipMemcpy(ry[b], by[b], sizeof(Tx) * size_y, hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                unit_check_general(1, N, batch_count, incx, hx_cpu, rx);
                unit_check_general(1, N, batch_count, incy, hy_cpu, ry);
            }
        }

        // Test device
        {
            for(int b = 0; b < batch_count; b++)
            {
                CHECK_HIP_ERROR(
                    hipMemcpy(bx[b], hx[b], sizeof(Tx) * size_x, hipMemcpyHostToDevice));
                CHECK_HIP_ERROR(
                    hipMemcpy(by[b], hy[b], sizeof(Tx) * size_y, hipMemcpyHostToDevice));
            }
            CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(Tx*) * batch_count, hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(dy, by, sizeof(Tx*) * batch_count, hipMemcpyHostToDevice));

            status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
            status_4 = (hipblasRotBatchedExFn(handle,
                                              N,
                                              dx,
                                              xType,
                                              incx,
                                              dy,
                                              yType,
                                              incy,
                                              dc,
                                              ds,
                                              csType,
                                              batch_count,
                                              executionType));

            if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
            {
                hipblasDestroy(handle);
                if(status_3 != HIPBLAS_STATUS_SUCCESS)
                    return status_3;
                if(status_4 != HIPBLAS_STATUS_SUCCESS)
                    return status_4;
            }

            host_vector<Tx> rx[batch_count];
            host_vector<Tx> ry[batch_count];
            for(int b = 0; b < batch_count; b++)
            {
                rx[b] = host_vector<Tx>(size_x);
                ry[b] = host_vector<Tx>(size_y);
                CHECK_HIP_ERROR(
                    hipMemcpy(rx[b], bx[b], sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
                CHECK_HIP_ERROR(
                    hipMemcpy(ry[b], by[b], sizeof(Tx) * size_y, hipMemcpyDeviceToHost));
            }

            if(arg.unit_check)
            {
                unit_check_general(1, N, batch_count, incx, hx_cpu, rx);
                unit_check_general(1, N, batch_count, incy, hy_cpu, ry);
            }
        }
    }

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_rot_batched_ex(Arguments arg)
{
    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16B
       && csType == HIPBLAS_R_16B)
    {
        status = testing_rot_batched_ex_template<float, hipblasBfloat16, hipblasBfloat16>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_16F
            && csType == HIPBLAS_R_16F)
    {
        status = testing_rot_batched_ex_template<float, hipblasHalf, hipblasHalf>(arg);
    }
    else if(executionType == HIPBLAS_R_32F && xType == yType && xType == HIPBLAS_R_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_batched_ex_template<float>(arg);
    }
    else if(executionType == HIPBLAS_R_64F && xType == yType && xType == HIPBLAS_R_64F
            && csType == HIPBLAS_R_64F)
    {
        status = testing_rot_batched_ex_template<double>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_R_32F)
    {
        status = testing_rot_batched_ex_template<hipblasComplex, hipblasComplex, float>(arg);
    }
    else if(executionType == HIPBLAS_C_32F && xType == yType && xType == HIPBLAS_C_32F
            && csType == HIPBLAS_C_32F)
    {
        status = testing_rot_batched_ex_template<hipblasComplex>(arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_R_64F)
    {
        status
            = testing_rot_batched_ex_template<hipblasDoubleComplex, hipblasDoubleComplex, double>(
                arg);
    }
    else if(executionType == HIPBLAS_C_64F && xType == yType && xType == HIPBLAS_C_64F
            && csType == HIPBLAS_C_64F)
    {
        status = testing_rot_batched_ex_template<hipblasDoubleComplex>(arg);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
