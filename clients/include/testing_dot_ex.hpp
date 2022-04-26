/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
hipblasStatus_t testing_dot_ex_template(const Arguments& argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasDotExFn = FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                  : (CONJ ? hipblasDotcEx : hipblasDotEx);

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(1);
        host_vector<Tr>   h_hipblas_result_0(1);
        hipblas_init_nan(h_hipblas_result_0.data(), 1);
        CHECK_HIP_ERROR(
            hipMemcpy(d_hipblas_result_0, h_hipblas_result_0, sizeof(Tr), hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           nullptr,
                                           xType,
                                           incx,
                                           nullptr,
                                           yType,
                                           incy,
                                           d_hipblas_result_0,
                                           resultType,
                                           executionType));

        host_vector<Tr> cpu_0(1);
        host_vector<Tr> gpu_0(1);
        CHECK_HIP_ERROR(hipMemcpy(gpu_0, d_hipblas_result_0, sizeof(Tr), hipMemcpyDeviceToHost));
        unit_check_general<Tr>(1, 1, 1, cpu_0, gpu_0);
        return HIPBLAS_STATUS_SUCCESS;
    }

    int    abs_incx = incx >= 0 ? incx : -incx;
    int    abs_incy = incy >= 0 ? incy : -incy;
    size_t sizeX    = size_t(N) * abs_incx;
    size_t sizeY    = size_t(N) * abs_incy;
    if(!sizeX)
        sizeX = 1;
    if(!sizeY)
        sizeY = 1;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Tr> d_hipblas_result(1);

    Tr cpu_result, hipblas_result_host, hipblas_result_device;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    hipblas_init_vector(hx, argus, N, abs_incx, 0, 1, hipblas_client_alpha_sets_nan, true, true);
    hipblas_init_vector(hy, argus, N, abs_incy, 0, 1, hipblas_client_alpha_sets_nan, false);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           &hipblas_result_host,
                                           resultType,
                                           executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           d_hipblas_result,
                                           resultType,
                                           executionType));

        CHECK_HIP_ERROR(
            hipMemcpy(&hipblas_result_device, d_hipblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? cblas_dotc<Tx> : cblas_dot<Tx>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(argus.unit_check)
        {
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_host);
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &hipblas_result_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_host);
            hipblas_error_device
                = norm_check_general<Tr>('F', 1, 1, 1, &cpu_result, &hipblas_result_device);
        }

    } // end of if unit/norm check

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

            CHECK_HIPBLAS_ERROR(hipblasDotExFn(handle,
                                               N,
                                               dx,
                                               xType,
                                               incx,
                                               dy,
                                               yType,
                                               incy,
                                               d_hipblas_result,
                                               resultType,
                                               executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy>{}.log_args<Tx>(std::cout,
                                                          argus,
                                                          gpu_time_used,
                                                          dot_gflop_count<CONJ, Tx>(N),
                                                          dot_gbyte_count<Tx>(N),
                                                          hipblas_error_host,
                                                          hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_dot_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_dot_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, hipblasHalf, false>(
            argus);
    }
    else if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        status
            = testing_dot_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, false>(argus);
    }
    else if(xType == HIPBLAS_R_16B && yType == HIPBLAS_R_16B && resultType == HIPBLAS_R_16B
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_ex_template<hipblasBfloat16,
                                         hipblasBfloat16,
                                         hipblasBfloat16,
                                         hipblasBfloat16,
                                         false>(argus);
    }
    else if(xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_ex_template<float, float, float, float, false>(argus);
    }
    else if(xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_dot_ex_template<double, double, double, double, false>(argus);
    }
    else if(xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F && resultType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_dot_ex_template<hipblasComplex,
                                         hipblasComplex,
                                         hipblasComplex,
                                         hipblasComplex,
                                         false>(argus);
    }
    else if(xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F && resultType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_dot_ex_template<hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         false>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}

hipblasStatus_t testing_dotc_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_dot_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, hipblasHalf, true>(
            argus);
    }
    else if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, true>(argus);
    }
    else if(xType == HIPBLAS_R_16B && yType == HIPBLAS_R_16B && resultType == HIPBLAS_R_16B
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_ex_template<hipblasBfloat16,
                                         hipblasBfloat16,
                                         hipblasBfloat16,
                                         hipblasBfloat16,
                                         true>(argus);
    }
    else if(xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_ex_template<float, float, float, float, true>(argus);
    }
    else if(xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_dot_ex_template<double, double, double, double, true>(argus);
    }
    else if(xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F && resultType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_dot_ex_template<hipblasComplex,
                                         hipblasComplex,
                                         hipblasComplex,
                                         hipblasComplex,
                                         true>(argus);
    }
    else if(xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F && resultType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_dot_ex_template<hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         hipblasDoubleComplex,
                                         true>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
