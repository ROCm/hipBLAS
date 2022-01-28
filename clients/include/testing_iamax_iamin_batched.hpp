/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

template <typename T>
using hipblas_iamax_iamin_batched_t = hipblasStatus_t (*)(
    hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

template <typename T, void REFBLAS_FUNC(int, const T*, int, int*)>
hipblasStatus_t testing_iamax_iamin_batched(const Arguments&                 argus,
                                            hipblas_iamax_iamin_batched_t<T> func)
{
    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;

    hipblasLocalHandle handle(argus);
    int                zero = 0;

    // check to prevent undefined memory allocation error
    if(batch_count <= 0 || N <= 0 || incx <= 0)
    {
        // quick return success
        device_vector<int> d_hipblas_result_0(std::max(1, batch_count));
        host_vector<int>   h_hipblas_result_0(std::max(1, batch_count));
        hipblas_init_nan(h_hipblas_result_0.data(), std::max(1, batch_count));
        CHECK_HIP_ERROR(hipMemcpy(d_hipblas_result_0,
                                  h_hipblas_result_0,
                                  sizeof(int) * std::max(1, batch_count),
                                  hipMemcpyHostToDevice));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(func(handle, N, nullptr, incx, batch_count, d_hipblas_result_0));

        if(batch_count > 0)
        {
            host_vector<int> cpu_0(batch_count);
            host_vector<int> gpu_0(batch_count);
            CHECK_HIP_ERROR(hipMemcpy(
                gpu_0, d_hipblas_result_0, sizeof(int) * batch_count, hipMemcpyDeviceToHost));
            unit_check_general<int>(1, batch_count, 1, cpu_0, gpu_0);
        }

        return HIPBLAS_STATUS_SUCCESS;
    }

    host_batch_vector<T> hx(N, incx, batch_count);
    host_vector<int>     cpu_result(batch_count);
    host_vector<int>     hipblas_result_host(batch_count);
    host_vector<int>     hipblas_result_device(batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_vector<int>     d_hipblas_result_device(batch_count);
    CHECK_HIP_ERROR(dx.memcheck());

    // Initial Data on CPU
    hipblas_init_vector(hx, argus, hipblas_client_alpha_sets_nan, true);
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    double gpu_time_used;
    int    hipblas_error_host = 0, hipblas_error_device = 0;

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
                    HIPBLAS
        =================================================================== */
        // device_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        CHECK_HIP_ERROR(hipMemcpy(hipblas_result_device,
                                  d_hipblas_result_device,
                                  sizeof(int) * batch_count,
                                  hipMemcpyDeviceToHost));

        // host_pointer
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(
            func(handle, N, dx.ptr_on_device(), incx, batch_count, hipblas_result_host));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            REFBLAS_FUNC(N, hx[b], incx, cpu_result + b);
            // change to Fortran 1 based indexing as in BLAS standard, not cblas zero based indexing
            cpu_result[b] += 1;
        }

        if(argus.unit_check)
        {
            unit_check_general<int>(1, 1, batch_count, cpu_result, hipblas_result_host);
            unit_check_general<int>(1, 1, batch_count, cpu_result, hipblas_result_device);
        }
        if(argus.norm_check)
        {
            for(int b = 0; b < batch_count; b++)
            {
                hipblas_error_host   = std::max(hipblas_error_host,
                                              std::abs(hipblas_result_host[b] - cpu_result[b]));
                hipblas_error_device = std::max(hipblas_error_device,
                                                std::abs(hipblas_result_device[b] - cpu_result[b]));
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

            CHECK_HIPBLAS_ERROR(
                func(handle, N, dx.ptr_on_device(), incx, batch_count, d_hipblas_result_device));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(std::cout,
                                                                argus,
                                                                gpu_time_used,
                                                                iamax_gflop_count<T>(N),
                                                                iamax_gbyte_count<T>(N),
                                                                hipblas_error_host,
                                                                hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}

template <typename T>
hipblasStatus_t testing_amax_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIamaxBatchedFn
        = FORTRAN ? hipblasIamaxBatched<T, true> : hipblasIamaxBatched<T, false>;

    return testing_iamax_iamin_batched<T, cblas_iamax<T>>(arg, hipblasIamaxBatchedFn);
}

template <typename T>
hipblasStatus_t testing_amin_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasIaminBatchedFn
        = FORTRAN ? hipblasIaminBatched<T, true> : hipblasIaminBatched<T, false>;

    return testing_iamax_iamin_batched<T, cblas_iamin<T>>(arg, hipblasIaminBatchedFn);
}
