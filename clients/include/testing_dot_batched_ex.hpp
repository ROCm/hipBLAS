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

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
hipblasStatus_t testing_dot_batched_ex_template(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasDotBatchedExFn
        = FORTRAN ? (CONJ ? hipblasDotcBatchedExFortran : hipblasDotBatchedExFortran)
                  : (CONJ ? hipblasDotcBatchedEx : hipblasDotBatchedEx);

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || incy < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<Tx> hx(N, incx, batch_count);
    host_batch_vector<Ty> hy(N, incy, batch_count);
    host_vector<Tr>       h_cpu_result(batch_count);
    host_vector<Tr>       h_hipblas_result_host(batch_count);
    host_vector<Tr>       h_hipblas_result_device(batch_count);

    device_batch_vector<Tx> dx(N, incx, batch_count);
    device_batch_vector<Ty> dy(N, incy, batch_count);
    device_vector<Tr>       d_hipblas_result(batch_count);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    hipblas_init(hy, true);
    hipblas_init_alternating_sign(hx);
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    /* =====================================================================
         HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                              N,
                                              dx.ptr_on_device(),
                                              xType,
                                              incx,
                                              dy.ptr_on_device(),
                                              yType,
                                              incy,
                                              batch_count,
                                              h_hipblas_result_host,
                                              resultType,
                                              executionType));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                              N,
                                              dx.ptr_on_device(),
                                              xType,
                                              incx,
                                              dy.ptr_on_device(),
                                              yType,
                                              incy,
                                              batch_count,
                                              d_hipblas_result,
                                              resultType,
                                              executionType));

    CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                              d_hipblas_result,
                              sizeof(Tr) * batch_count,
                              hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            (CONJ ? cblas_dotc<Tx>
                  : cblas_dot<Tx>)(N, hx[b], incx, hy[b], incy, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
            unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_host);
            hipblas_error_device = norm_check_general<Tr>(
                'F', 1, batch_count, 1, h_cpu_result, h_hipblas_result_device);
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

            CHECK_HIPBLAS_ERROR(hipblasDotBatchedExFn(handle,
                                                      N,
                                                      dx.ptr_on_device(),
                                                      xType,
                                                      incx,
                                                      dy.ptr_on_device(),
                                                      yType,
                                                      incy,
                                                      batch_count,
                                                      d_hipblas_result,
                                                      resultType,
                                                      executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<Tx>(
            std::cout,
            argus,
            gpu_time_used,
            dot_gflop_count<CONJ, Tx>(N),
            dot_gbyte_count<Tx>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_dot_batched_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_dot_batched_ex_template<hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 false>(argus);
    }
    else if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        status
            = testing_dot_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, false>(
                argus);
    }
    else if(xType == HIPBLAS_R_16B && yType == HIPBLAS_R_16B && resultType == HIPBLAS_R_16B
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_batched_ex_template<hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 false>(argus);
    }
    else if(xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_batched_ex_template<float, float, float, float, false>(argus);
    }
    else if(xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_dot_batched_ex_template<double, double, double, double, false>(argus);
    }
    else if(xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F && resultType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_dot_batched_ex_template<hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 false>(argus);
    }
    else if(xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F && resultType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_dot_batched_ex_template<hipblasDoubleComplex,
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

hipblasStatus_t testing_dotc_batched_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t yType         = argus.b_type;
    hipblasDatatype_t resultType    = argus.c_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
       && executionType == HIPBLAS_R_16F)
    {
        status = testing_dot_batched_ex_template<hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 hipblasHalf,
                                                 true>(argus);
    }
    else if(xType == HIPBLAS_R_16F && yType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F
            && executionType == HIPBLAS_R_32F)
    {
        status
            = testing_dot_batched_ex_template<hipblasHalf, hipblasHalf, hipblasHalf, float, true>(
                argus);
    }
    else if(xType == HIPBLAS_R_16B && yType == HIPBLAS_R_16B && resultType == HIPBLAS_R_16B
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_batched_ex_template<hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 hipblasBfloat16,
                                                 true>(argus);
    }
    else if(xType == HIPBLAS_R_32F && yType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F
            && executionType == HIPBLAS_R_32F)
    {
        status = testing_dot_batched_ex_template<float, float, float, float, true>(argus);
    }
    else if(xType == HIPBLAS_R_64F && yType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F
            && executionType == HIPBLAS_R_64F)
    {
        status = testing_dot_batched_ex_template<double, double, double, double, true>(argus);
    }
    else if(xType == HIPBLAS_C_32F && yType == HIPBLAS_C_32F && resultType == HIPBLAS_C_32F
            && executionType == HIPBLAS_C_32F)
    {
        status = testing_dot_batched_ex_template<hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 hipblasComplex,
                                                 true>(argus);
    }
    else if(xType == HIPBLAS_C_64F && yType == HIPBLAS_C_64F && resultType == HIPBLAS_C_64F
            && executionType == HIPBLAS_C_64F)
    {
        status = testing_dot_batched_ex_template<hipblasDoubleComplex,
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
