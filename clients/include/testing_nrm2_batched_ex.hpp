/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
hipblasStatus_t testing_nrm2_batched_ex_template(Arguments argus)
{
    bool FORTRAN                = argus.fortran;
    auto hipblasNrm2BatchedExFn = FORTRAN ? hipblasNrm2BatchedExFortran : hipblasNrm2BatchedEx;

    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;

    // check to prevent undefined memory allocation error
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(!batch_count)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t resultType    = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    int sizeX = N * incx;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx[batch_count];
    host_vector<Tr> h_cpu_result(batch_count);
    host_vector<Tr> h_rocblas_result1(batch_count);
    host_vector<Tr> h_rocblas_result2(batch_count);

    device_batch_vector<Tx> bx(batch_count, sizeX);

    device_vector<Tx*, 0, Tx> dx(batch_count);
    device_vector<Tr>         d_rocblas_result(batch_count);

    int last = batch_count - 1;
    if(!dx || !d_rocblas_result || (!bx[last] && sizeX))
    {
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hx[b] = host_vector<Tx>(sizeX);

        srand(1);
        hipblas_init<Tx>(hx[b], 1, N, incx);

        CHECK_HIP_ERROR(hipMemcpy(bx[b], hx[b], sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    }
    CHECK_HIP_ERROR(hipMemcpy(dx, bx, sizeof(Tx*) * batch_count, hipMemcpyHostToDevice));

    // hipblasNrm2 accept both dev/host pointer for the scalar
    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    status_2 = hipblasNrm2BatchedExFn(
        handle, N, dx, xType, incx, batch_count, d_rocblas_result, resultType, executionType);

    status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    status_4 = hipblasNrm2BatchedExFn(
        handle, N, dx, xType, incx, batch_count, h_rocblas_result2, resultType, executionType);

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

    CHECK_HIP_ERROR(hipMemcpy(
        h_rocblas_result1, d_rocblas_result, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_nrm2<Tx, Tr>(N, hx[b], incx, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_rocblas_result1, N);
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_rocblas_result2, N);
        }

    } // end of if unit/norm check

    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t testing_nrm2_batched_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t resultType    = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_batched_ex_template<hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(xType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_batched_ex_template<float>(argus);
    }
    else if(xType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_nrm2_batched_ex_template<double>(argus);
    }
    else if(xType == HIPBLAS_C_32F && resultType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_batched_ex_template<hipblasComplex, float>(argus);
    }
    else if(xType == HIPBLAS_C_64F && resultType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_nrm2_batched_ex_template<hipblasDoubleComplex, double>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
