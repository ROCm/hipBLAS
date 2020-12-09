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
hipblasStatus_t testing_nrm2_strided_batched_ex_template(Arguments argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasNrm2StridedBatchedExFn
        = FORTRAN ? hipblasNrm2StridedBatchedExFortran : hipblasNrm2StridedBatchedEx;

    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    int stridex = N * incx * stride_scale;
    int sizeX   = stridex * batch_count;

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

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Tr> h_rocblas_result1(batch_count);
    host_vector<Tr> h_rocblas_result2(batch_count);
    host_vector<Tr> h_cpu_result(batch_count);

    device_vector<Tx> dx(sizeX);
    device_vector<Tr> d_rocblas_result(batch_count);

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Tx>(hx, 1, N, incx, stridex, batch_count);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

    // hipblasNrm2 accept both dev/host pointer for the scalar
    status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
    status_2 = hipblasNrm2StridedBatchedExFn(handle,
                                             N,
                                             dx,
                                             xType,
                                             incx,
                                             stridex,
                                             batch_count,
                                             d_rocblas_result,
                                             resultType,
                                             executionType);

    status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    status_4 = hipblasNrm2StridedBatchedExFn(handle,
                                             N,
                                             dx,
                                             xType,
                                             incx,
                                             stridex,
                                             batch_count,
                                             h_rocblas_result2,
                                             resultType,
                                             executionType);

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
            cblas_nrm2<Tx, Tr>(N, hx.data() + b * stridex, incx, &(h_cpu_result[b]));
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

hipblasStatus_t testing_nrm2_strided_batched_ex(Arguments argus)
{
    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t resultType    = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(xType == HIPBLAS_R_16F && resultType == HIPBLAS_R_16F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_strided_batched_ex_template<hipblasHalf, hipblasHalf, float>(argus);
    }
    else if(xType == HIPBLAS_R_32F && resultType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_strided_batched_ex_template<float>(argus);
    }
    else if(xType == HIPBLAS_R_64F && resultType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_nrm2_strided_batched_ex_template<double>(argus);
    }
    else if(xType == HIPBLAS_C_32F && resultType == HIPBLAS_R_32F && executionType == HIPBLAS_R_32F)
    {
        status = testing_nrm2_strided_batched_ex_template<hipblasComplex, float>(argus);
    }
    else if(xType == HIPBLAS_C_64F && resultType == HIPBLAS_R_64F && executionType == HIPBLAS_R_64F)
    {
        status = testing_nrm2_strided_batched_ex_template<hipblasDoubleComplex, double>(argus);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
