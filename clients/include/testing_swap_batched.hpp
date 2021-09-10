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
hipblasStatus_t testing_swap_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasSwapBatchedFn
        = FORTRAN ? hipblasSwapBatched<T, true> : hipblasSwapBatched<T, false>;

    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int batch_count = argus.batch_count;
    int unit_check  = argus.unit_check;
    int norm_check  = argus.norm_check;
    int timing      = argus.timing;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N <= 0 || batch_count <= 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    int abs_incx = incx >= 0 ? incx : -incx;
    int abs_incy = incy >= 0 ? incy : -incy;

    double hipblas_error = 0.0;
    double gpu_time_used = 0.0;

    hipblasLocalHandle handle(argus);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    host_batch_vector<T> hx(N, incx ? incx : 1, batch_count);
    host_batch_vector<T> hy(N, incy ? incy : 1, batch_count);
    host_batch_vector<T> hx_cpu(N, incx ? incx : 1, batch_count);
    host_batch_vector<T> hy_cpu(N, incy ? incy : 1, batch_count);

    device_batch_vector<T> dx(N, incx ? incx : 1, batch_count);
    device_batch_vector<T> dy(N, incy ? incy : 1, batch_count);

    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    // Initial Data on CPU
    hipblas_init(hx, true);
    hipblas_init(hy, false);
    hx_cpu.copy_from(hx);
    hy_cpu.copy_from(hy);
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));

    if(unit_check || norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSwapBatchedFn(
            handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));

        CHECK_HIP_ERROR(hx.transfer_from(dx));
        CHECK_HIP_ERROR(hy.transfer_from(dy));

        /* =====================================================================
                    CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_swap<T>(N, hx_cpu[b], incx, hy_cpu[b], incy);
        }

        if(unit_check)
        {
            unit_check_general<T>(1, N, batch_count, abs_incx, hx_cpu, hx);
            unit_check_general<T>(1, N, batch_count, abs_incy, hy_cpu, hy);
        }
        if(norm_check)
        {
            hipblas_error
                = std::max(norm_check_general<T>('F', 1, N, abs_incx, hx_cpu, hx, batch_count),
                           norm_check_general<T>('F', 1, N, abs_incy, hy_cpu, hy, batch_count));
        }

    } // end of if unit/norm check

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasSwapBatchedFn(
                handle, N, dx.ptr_on_device(), incx, dy.ptr_on_device(), incy, batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.log_args<T>(std::cout,
                                                                        argus,
                                                                        gpu_time_used,
                                                                        swap_gflop_count<T>(N),
                                                                        swap_gbyte_count<T>(N),
                                                                        hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
