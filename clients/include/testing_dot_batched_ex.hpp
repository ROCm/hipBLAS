/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "hipblas.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename Tx, typename Ty = Tx, typename Tr = Ty, typename Tex = Tr, bool CONJ = false>
hipblasStatus_t testing_dot_batched_ex_template(Arguments argus)
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

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;

    int sizeX = N * incx;
    int sizeY = N * incy;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx[batch_count];
    host_vector<Ty> hy[batch_count];
    host_vector<Tr> h_cpu_result(batch_count);
    host_vector<Tr> h_rocblas_result1(batch_count);
    host_vector<Tr> h_rocblas_result2(batch_count);

    device_batch_vector<Tx> bx(batch_count, sizeX);
    device_batch_vector<Ty> by(batch_count, sizeY);

    device_vector<Tx*, 0, Tx> dx(batch_count);
    device_vector<Ty*, 0, Ty> dy(batch_count);
    device_vector<Tr>         d_rocblas_result(batch_count);

    int last = batch_count - 1;
    if(!dx || !dy || !d_rocblas_result || (!bx[last] && sizeX) || (!by[last] && sizeY))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    int device_pointer = 1;
    int host_pointer   = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx[b] = host_vector<Tx>(sizeX);
        hy[b] = host_vector<Ty>(sizeY);

        srand(1);
        hipblas_init_alternating_sign<Tx>(hx[b], 1, N, incx);
        hipblas_init<Ty>(hy[b], 1, N, incy);

        CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(by[b], hy[b], sizeof(Ty) * sizeY, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx, bx, batch_count * sizeof(Tx*), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, by, batch_count * sizeof(Ty*), hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    if(host_pointer)
    {
        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_1 = (hipblasDotBatchedExFn)(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           batch_count,
                                           d_rocblas_result,
                                           resultType,
                                           executionType);
    }
    if(device_pointer)
    {

        hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        status_2 = (hipblasDotBatchedExFn)(handle,
                                           N,
                                           dx,
                                           xType,
                                           incx,
                                           dy,
                                           yType,
                                           incy,
                                           batch_count,
                                           h_rocblas_result2,
                                           resultType,
                                           executionType);
    }

    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
    {
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
    }

    if(device_pointer)
        CHECK_HIP_ERROR(hipMemcpy(
            h_rocblas_result1, d_rocblas_result, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));

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
            unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_rocblas_result1);
            unit_check_general<Tr>(1, batch_count, 1, h_cpu_result, h_rocblas_result2);
        }

    } // end of if unit/norm check

    hipblasDestroy(handle);
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
