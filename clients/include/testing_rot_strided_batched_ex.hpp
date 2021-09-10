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

template <typename Tex, typename Tx = Tex, typename Tcs = Tx>
hipblasStatus_t testing_rot_strided_batched_ex_template(const Arguments& arg)
{
    using Ty     = Tx;
    bool FORTRAN = arg.fortran;
    auto hipblasRotStridedBatchedExFn
        = FORTRAN ? hipblasRotStridedBatchedExFortran : hipblasRotStridedBatchedEx;

    int    N            = arg.N;
    int    incx         = arg.incx;
    int    incy         = arg.incy;
    double stride_scale = arg.stride_scale;
    int    batch_count  = arg.batch_count;

    int           abs_incx = incx >= 0 ? incx : -incx;
    int           abs_incy = incy >= 0 ? incy : -incy;
    hipblasStride stridex  = N * abs_incx * stride_scale;
    hipblasStride stridey  = N * abs_incy * stride_scale;

    size_t size_x = stridex * batch_count;
    size_t size_y = stridey * batch_count;
    if(!size_x)
        size_x = 1;
    if(!size_y)
        size_y = 1;

    // check to prevent undefined memory allocation error
    if(N <= 0 || batch_count <= 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t xType         = arg.a_type;
    hipblasDatatype_t yType         = arg.b_type;
    hipblasDatatype_t csType        = arg.c_type;
    hipblasDatatype_t executionType = arg.compute_type;

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(arg);

    device_vector<Tx>  dx(size_x);
    device_vector<Ty>  dy(size_y);
    device_vector<Tcs> dc(1);
    device_vector<Tcs> ds(1);

    // Initial Data on CPU
    host_vector<Tx>  hx_host(size_x);
    host_vector<Ty>  hy_host(size_y);
    host_vector<Tx>  hx_device(size_x);
    host_vector<Ty>  hy_device(size_y);
    host_vector<Tx>  hx_cpu(size_x);
    host_vector<Ty>  hy_cpu(size_y);
    host_vector<Tcs> hc(1);
    host_vector<Tcs> hs(1);
    srand(1);
    hipblas_init<Tx>(hx_host, 1, N, abs_incx, stridex, batch_count);
    hipblas_init<Ty>(hy_host, 1, N, abs_incy, stridey, batch_count);

    hipblas_init<Tcs>(hc, 1, 1, 1);
    hipblas_init<Tcs>(hs, 1, 1, 1);

    hx_cpu = hx_device = hx_host;
    hy_cpu = hy_device = hy_host;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx_host, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy_host, sizeof(Ty) * size_y, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(Tcs), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(Tcs), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasRotStridedBatchedExFn(handle,
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

        CHECK_HIP_ERROR(hipMemcpy(hx_host, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_host, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(dx, hx_device, sizeof(Tx) * size_x, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy, hy_device, sizeof(Ty) * size_y, hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasRotStridedBatchedExFn(handle,
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

        CHECK_HIP_ERROR(hipMemcpy(hx_device, dx, sizeof(Tx) * size_x, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_device, dy, sizeof(Ty) * size_y, hipMemcpyDeviceToHost));

        for(int b = 0; b < batch_count; b++)
        {
            cblas_rot<Tx, Tcs, Tcs>(
                N, hx_cpu.data() + b * stridex, incx, hy_cpu.data() + b * stridey, incy, *hc, *hs);
        }

        if(arg.unit_check)
        {
            unit_check_general<Tx>(1, N, batch_count, abs_incx, stridex, hx_cpu, hx_host);
            unit_check_general<Tx>(1, N, batch_count, abs_incy, stridey, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, batch_count, abs_incx, stridex, hx_cpu, hx_device);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, stridey, hy_cpu, hy_device);
        }

        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<Tx>(
                'F', 1, N, abs_incx, stridex, hx_cpu, hx_host, batch_count);
            hipblas_error_host += norm_check_general<Ty>(
                'F', 1, N, abs_incy, stridey, hy_cpu, hy_host, batch_count);
            hipblas_error_device = norm_check_general<Tx>(
                'F', 1, N, abs_incx, stridex, hx_cpu, hx_device, batch_count);
            hipblas_error_device += norm_check_general<Ty>(
                'F', 1, N, abs_incy, stridey, hy_cpu, hy_device, batch_count);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasRotStridedBatchedExFn(handle,
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
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_incy, e_stride_y>{}.log_args<Tx>(
            std::cout,
            arg,
            gpu_time_used,
            rot_gflop_count<Tx, Ty, Tcs, Tcs>(N),
            rot_gbyte_count<Tx>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

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
