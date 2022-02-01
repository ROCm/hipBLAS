/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename Ta, typename Tx = Ta, typename Ty = Tx>
hipblasStatus_t testing_axpy_batched_ex_template(const Arguments& argus)
{
    bool FORTRAN                = argus.fortran;
    auto hipblasAxpyBatchedExFn = FORTRAN ? hipblasAxpyBatchedExFortran : hipblasAxpyBatchedEx;

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasAxpyBatchedExFn(handle,
                                                   N,
                                                   nullptr,
                                                   alphaType,
                                                   nullptr,
                                                   xType,
                                                   incx,
                                                   nullptr,
                                                   yType,
                                                   incy,
                                                   batch_count,
                                                   executionType));
        return HIPBLAS_STATUS_SUCCESS;
    }

    int abs_incy = incy < 0 ? -incy : incy;

    Ta h_alpha = argus.get_alpha<Ta>();

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy_host(N, incy, batch_count);
    host_batch_vector<Ty> hy_device(N, incy, batch_count);
    host_batch_vector<Ty> hy_cpu(N, incy, batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Ta>       d_alpha(1);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, argus, hipblas_client_alpha_sets_nan, true);
    hipblas_init_vector(hy_host, argus, hipblas_client_alpha_sets_nan, false);

    hy_device.copy_from(hy_host);
    hy_cpu.copy_from(hy_host);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(Ta), hipMemcpyHostToDevice));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasAxpyBatchedExFn(handle,
                                               N,
                                               &h_alpha,
                                               alphaType,
                                               dx.ptr_on_device(),
                                               xType,
                                               incx,
                                               dy.ptr_on_device(),
                                               yType,
                                               incy,
                                               batch_count,
                                               executionType));

    CHECK_HIP_ERROR(hy_host.transfer_from(dy));
    CHECK_HIP_ERROR(dy.transfer_from(hy_device));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasAxpyBatchedExFn(handle,
                                               N,
                                               d_alpha,
                                               alphaType,
                                               dx.ptr_on_device(),
                                               xType,
                                               incx,
                                               dy.ptr_on_device(),
                                               yType,
                                               incy,
                                               batch_count,
                                               executionType));

    CHECK_HIP_ERROR(hy_device.transfer_from(dy));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_axpy(N, h_alpha, hx[b], incx, hy_cpu[b], incy);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_host);
            unit_check_general<Ty>(1, N, batch_count, abs_incy, hy_cpu, hy_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_host, batch_count);
            hipblas_error_device
                = norm_check_general<Ty>('F', 1, N, abs_incy, hy_cpu, hy_device, batch_count);
        }

    } // end of if unit check

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasAxpyBatchedExFn(handle,
                                                       N,
                                                       d_alpha,
                                                       alphaType,
                                                       dx.ptr_on_device(),
                                                       xType,
                                                       incx,
                                                       dy.ptr_on_device(),
                                                       yType,
                                                       incy,
                                                       batch_count,
                                                       executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<Ta>(std::cout,
                                                                         argus,
                                                                         gpu_time_used,
                                                                         axpy_gflop_count<Ta>(N),
                                                                         axpy_gbyte_count<Ta>(N),
                                                                         hipblas_error_host,
                                                                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_axpy_batched_ex(Arguments argus)
{
    hipblasDatatype_t alphaType     = argus.a_type;
    hipblasDatatype_t xType         = argus.b_type;
    hipblasDatatype_t yType         = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_axpy_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_16F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_batched_ex_template<hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        // Not testing accumulation here
        status = testing_axpy_batched_ex_template<float, hipblasHalf>(argus);
    }
    else if(alphaType == HIPBLAS_R_32F && xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_axpy_batched_ex_template<float>(argus);
    }
    else if(alphaType == HIPBLAS_R_64F && xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_axpy_batched_ex_template<double>(argus);
    }
    else if(alphaType == HIPBLAS_C_32F && xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_axpy_batched_ex_template<hipblasComplex>(argus);
    }
    else if(alphaType == HIPBLAS_C_64F && xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_axpy_batched_ex_template<hipblasDoubleComplex>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
