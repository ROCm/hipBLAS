/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename Ta, typename Tx = Ta, typename Ty = Tx>
hipblasStatus_t testing_axpy_strided_batched_ex_template(Arguments argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasAxpyStridedBatchedExFn
        = FORTRAN ? hipblasAxpyStridedBatchedExFortran : hipblasAxpyStridedBatchedEx;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int    N            = argus.N;
    int    incx         = argus.incx;
    int    incy         = argus.incy;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || !incx || !incy || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    int abs_incx = incx < 0 ? -incx : incx;
    int abs_incy = incy < 0 ? -incy : incy;

    hipblasStride stridex = N * abs_incx * stride_scale;
    hipblasStride stridey = N * abs_incy * stride_scale;

    int sizeX = stridex * batch_count;
    int sizeY = stridey * batch_count;
    Ta  alpha = argus.alpha;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy(sizeY);
    host_vector<Tx> hx_cpu(sizeX);
    host_vector<Ty> hy_cpu(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);

    double gpu_time_used, cpu_time_used;
    double rocblas_error = 0.0;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Tx>(hx, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<Ty>(hy, 1, N, abs_incy, stridey, batch_count);

    hx_cpu = hx;
    hy_cpu = hy;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    status = hipblasAxpyStridedBatchedExFn(handle,
                                           N,
                                           &alpha,
                                           alphaType,
                                           dx,
                                           xType,
                                           incx,
                                           stridex,
                                           dy,
                                           yType,
                                           incy,
                                           stridey,
                                           batch_count,
                                           executionType);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblasDestroy(handle);
        return status;
    }

    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(Tx) * sizeX, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(Ty) * sizeY, hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy<Tx>(N, alpha, hx_cpu.data() + b * stridex, incx, hy_cpu + b * stridey, incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, abs_incx, stridex, hx_cpu.data(), hx.data());
            unit_check_general<Ty>(1, N, batch_count, abs_incy, stridey, hy_cpu.data(), hy.data());
        }

    } // end of if unit check

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_axpy_strided_batched_ex(Arguments argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_axpy_strided_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_strided_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_strided_batched_ex_template<float, hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_strided_batched_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_axpy_strided_batched_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_axpy_strided_batched_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_axpy_strided_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
