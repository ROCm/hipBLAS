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

template <typename Tx, typename Tr = Tx, typename Tex = Tr>
hipblasStatus_t testing_nrm2_strided_batched_ex_template(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasNrm2StridedBatchedExFn
        = FORTRAN ? hipblasNrm2StridedBatchedExFortran : hipblasNrm2StridedBatchedEx;

    int    N            = argus.N;
    int    incx         = argus.incx;
    double stride_scale = argus.stride_scale;
    int    batch_count  = argus.batch_count;

    hipblasStride stridex = size_t(N) * incx * stride_scale;
    size_t        sizeX   = stridex * batch_count;

    hipblasDatatype_t xType         = argus.a_type;
    hipblasDatatype_t resultType    = argus.b_type;
    hipblasDatatype_t executionType = argus.compute_type;

    hipblasLocalHandle handle(argus);

    // check to prevent undefined memory allocation error
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        device_vector<Tr> d_hipblas_result_0(std::max(batch_count, 1));
        host_vector<Tr>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(d_hipblas_result_0,
                                  h_hipblas_result_0,
                                  sizeof(Tr) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFortran(handle,
                                                               N,
                                                               nullptr,
                                                               xType,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               d_hipblas_result_0,
                                                               resultType,
                                                               executionType));

        if(batch_count > 0)
        {
            // TODO: error in rocBLAS - only setting the first element to 0, not for all batches
            // host_vector<Tr> cpu_0(batch_count);
            // host_vector<Tr> gpu_0(batch_count);
            // CHECK_HIP_ERROR(hipMemcpy(
            //     gpu_0, d_hipblas_result_0, sizeof(Tr) * batch_count, hipMemcpyDeviceToHost));
            // unit_check_general<Tr>(1, batch_count, 1, cpu_0, gpu_0);
        }
        return HIPBLAS_STATUS_SUCCESS;
    }

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_vector<Tx> hx(sizeX);
    host_vector<Tr> h_hipblas_result_host(batch_count);
    host_vector<Tr> h_hipblas_result_device(batch_count);
    host_vector<Tr> h_cpu_result(batch_count);

    device_vector<Tx> dx(sizeX);
    device_vector<Tr> d_hipblas_result(batch_count);

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    // Initial Data on CPU
    srand(1);
    hipblas_init<Tx>(hx, 1, N, incx, stridex, batch_count);

    // copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx, sizeof(Tx) * sizeX, hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                          N,
                                                          dx,
                                                          xType,
                                                          incx,
                                                          stridex,
                                                          batch_count,
                                                          d_hipblas_result,
                                                          resultType,
                                                          executionType));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                          N,
                                                          dx,
                                                          xType,
                                                          incx,
                                                          stridex,
                                                          batch_count,
                                                          h_hipblas_result_host,
                                                          resultType,
                                                          executionType));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_nrm2<Tx, Tr>(N, hx.data() + b * stridex, incx, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_nrm2<Tr, Tex>(batch_count, h_cpu_result, h_hipblas_result_host, N);
            unit_check_nrm2<Tr, Tex>(batch_count, h_cpu_result, h_hipblas_result_device, N);
        }
        if(argus.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                hipblas_error_host
                    = std::max(vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_host[b])),
                               hipblas_error_host);
                hipblas_error_device = std::max(
                    vector_norm_1(1, 1, &(h_cpu_result[b]), &(h_hipblas_result_device[b])),
                    hipblas_error_device);
            }
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

            CHECK_HIPBLAS_ERROR(hipblasNrm2StridedBatchedExFn(handle,
                                                              N,
                                                              dx,
                                                              xType,
                                                              incx,
                                                              stridex,
                                                              batch_count,
                                                              d_hipblas_result,
                                                              resultType,
                                                              executionType));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_stride_x, e_batch_count>{}.log_args<Tx>(
            std::cout,
            argus,
            gpu_time_used,
            nrm2_gflop_count<Tx>(N),
            nrm2_gbyte_count<Tx>(N),
            hipblas_error_host,
            hipblas_error_device);
    }

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
