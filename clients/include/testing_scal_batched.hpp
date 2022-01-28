/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, typename U = T>
hipblasStatus_t testing_scal_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasScalBatchedFn
        = FORTRAN ? hipblasScalBatched<T, U, true> : hipblasScalBatched<T, U, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int batch_count = argus.batch_count;
    int unit_check  = argus.unit_check;
    int norm_check  = argus.norm_check;
    int timing      = argus.timing;

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || incx <= 0 || batch_count <= 0)
    {
        CHECK_HIPBLAS_ERROR(hipblasScalBatchedFn(handle, N, nullptr, nullptr, incx, batch_count));
        return HIPBLAS_STATUS_SUCCESS;
    }

    size_t sizeX         = size_t(N) * incx;
    U      alpha         = argus.get_alpha<U>();
    double gpu_time_used = 0.0, cpu_time_used = 0.0;
    double hipblas_error = 0.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx, batch_count);
    host_batch_vector<T> hz(N, incx, batch_count);

    device_batch_vector<T> dx(N, incx, batch_count);
    device_batch_vector<T> dz(N, incx, batch_count);
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dz.memcheck());

    hipblas_init_vector(hx, argus, hipblas_client_alpha_sets_nan, true);
    hz.copy_from(hx);

    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dz.transfer_from(hx));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(
            hipblasScalBatchedFn(handle, N, &alpha, dx.ptr_on_device(), incx, batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx.transfer_from(dx));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_scal<T, U>(N, alpha, hz[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(unit_check)
        {
            unit_check_general<T>(1, N, batch_count, incx, hz, hx);
        }
        if(norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, N, incx, hz, hx, batch_count);
        }

    } // end of if unit check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(
                hipblasScalBatchedFn(handle, N, &alpha, dx.ptr_on_device(), incx, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_batch_count>{}.log_args<T>(std::cout,
                                                                argus,
                                                                gpu_time_used,
                                                                scal_gflop_count<T, U>(N),
                                                                scal_gbyte_count<T>(N),
                                                                hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
