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

template <typename T>
hipblasStatus_t testing_nrm2_batched(const Arguments& argus)
{
    using Tr     = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasNrm2BatchedFn
        = FORTRAN ? hipblasNrm2Batched<T, Tr, true> : hipblasNrm2Batched<T, Tr, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;

    // check to prevent undefined memory allocation error
    if(N < 0 || incx < 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int sizeX = N * incx;

    double gpu_time_used;
    double hipblas_error_host = 0, hipblas_error_device = 0;

    hipblasLocalHandle handle(argus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<Tr>      h_cpu_result(batch_count);
    host_vector<Tr>      h_hipblas_result_host(batch_count);
    host_vector<Tr>      h_hipblas_result_device(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<Tr>      d_hipblas_result(batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    // Initial Data on CPU
    hipblas_init(hx, true);
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    if(argus.unit_check || argus.norm_check)
    {
        // hipblasNrm2 accept both dev/host pointer for the scalar
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasNrm2BatchedFn(
            handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasNrm2BatchedFn(
            handle, N, dx.ptr_on_device(), incx, batch_count, h_hipblas_result_host));

        CHECK_HIP_ERROR(hipMemcpy(h_hipblas_result_device,
                                  d_hipblas_result,
                                  sizeof(Tr) * batch_count,
                                  hipMemcpyDeviceToHost));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_nrm2<T, Tr>(N, hx[b], incx, &(h_cpu_result[b]));
        }

        if(argus.unit_check)
        {
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_hipblas_result_host, N);
            unit_check_nrm2<Tr>(batch_count, h_cpu_result, h_hipblas_result_device, N);
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

            CHECK_HIPBLAS_ERROR(hipblasNrm2BatchedFn(
                handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(std::cout,
                                                                argus,
                                                                gpu_time_used,
                                                                nrm2_gflop_count<T>(N),
                                                                nrm2_gbyte_count<T>(N),
                                                                hipblas_error_host,
                                                                hipblas_error_device);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
