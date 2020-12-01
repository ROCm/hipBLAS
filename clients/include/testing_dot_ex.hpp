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
hipblasStatus_t testing_dot_ex_template(Arguments argus)
{
    bool FORTRAN        = argus.fortran;
    auto hipblasDotExFn = FORTRAN ? (CONJ ? hipblasDotcExFortran : hipblasDotExFortran)
                                  : (CONJ ? hipblasDotcEx : hipblasDotEx);

    int N    = argus.N;
    int incx = argus.incx;
    int incy = argus.incy;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || incx < 0 || incy < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
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
    host_vector<Tx> hx(sizeX);
    host_vector<Ty> hy(sizeY);

    device_vector<Tx> dx(sizeX);
    device_vector<Ty> dy(sizeY);
    device_vector<Tr> d_rocblas_result(1);

    Tr  cpu_result, rocblas_result;
    int device_pointer = 1;

    double gpu_time_used, cpu_time_used;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init_alternating_sign<Tx>(hx, 1, N, incx);
    hipblas_init<Ty>(hy, 1, N, incy);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(Tx) * sizeX, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(Ty) * sizeY, hipMemcpyHostToDevice));

    /* =====================================================================
         ROCBLAS
    =================================================================== */
    /* =====================================================================
                CPU BLAS
    =================================================================== */
    // hipblasDot accept both dev/host pointer for the scalar
    if(device_pointer)
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);

        status_2 = (hipblasDotExFn)(handle,
                                    N,
                                    dx,
                                    xType,
                                    incx,
                                    dy,
                                    yType,
                                    incy,
                                    d_rocblas_result,
                                    resultType,
                                    executionType);
    }
    else
    {

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = (hipblasDotExFn)(handle,
                                    N,
                                    dx,
                                    xType,
                                    incx,
                                    dy,
                                    yType,
                                    incy,
                                    &rocblas_result,
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
        CHECK_HIP_ERROR(
            hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(Tr), hipMemcpyDeviceToHost));

    if(argus.unit_check || argus.norm_check)
    {

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        (CONJ ? cblas_dotc<Tx> : cblas_dot<Tx>)(N, hx.data(), incx, hy.data(), incy, &cpu_result);

        if(argus.unit_check)
        {
            unit_check_general<Tr>(1, 1, 1, &cpu_result, &rocblas_result);
        }

    } // end of if unit/norm check

    hipblasDestroy(handle);
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
