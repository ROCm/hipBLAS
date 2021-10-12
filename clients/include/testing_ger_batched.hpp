/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, bool CONJ>
hipblasStatus_t testing_ger_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasGerBatchedFn
        = FORTRAN ? (CONJ ? hipblasGerBatched<T, true, true> : hipblasGerBatched<T, false, true>)
                  : (CONJ ? hipblasGerBatched<T, true, false> : hipblasGerBatched<T, false, false>);

    int M           = argus.M;
    int N           = argus.N;
    int incx        = argus.incx;
    int incy        = argus.incy;
    int lda         = argus.lda;
    int batch_count = argus.batch_count;

    size_t A_size = size_t(lda) * N;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    T h_alpha = argus.get_alpha<T>();

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || N < 0 || !incx || !incy || lda < M || lda < 1 || batch_count < 0;
    if(invalid_size || !M || !N || !batch_count)
    {
        hipblasStatus_t actual = hipblasGerBatchedFn(
            handle, M, N, nullptr, nullptr, incx, nullptr, incy, nullptr, lda, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hA_cpu(A_size, 1, batch_count);
    host_batch_vector<T> hA_host(A_size, 1, batch_count);
    host_batch_vector<T> hA_device(A_size, 1, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hy(N, incy, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);
    device_batch_vector<T> dy(N, incy, batch_count);
    device_vector<T>       d_alpha(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());
    CHECK_HIP_ERROR(dy.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hx);
    hipblas_init(hy);

    hA_cpu.copy_from(hA);
    hA_host.copy_from(hA);
    hA_device.copy_from(hA);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));
    CHECK_HIP_ERROR(dy.transfer_from(hy));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
            HIPBLAS
        =================================================================== */
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR(hipblasGerBatchedFn(handle,
                                                M,
                                                N,
                                                (T*)&h_alpha,
                                                dx.ptr_on_device(),
                                                incx,
                                                dy.ptr_on_device(),
                                                incy,
                                                dA.ptr_on_device(),
                                                lda,
                                                batch_count));

        CHECK_HIP_ERROR(hA_host.transfer_from(dA));
        CHECK_HIP_ERROR(dA.transfer_from(hA));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR(hipblasGerBatchedFn(handle,
                                                M,
                                                N,
                                                d_alpha,
                                                dx.ptr_on_device(),
                                                incx,
                                                dy.ptr_on_device(),
                                                incy,
                                                dA.ptr_on_device(),
                                                lda,
                                                batch_count));

        CHECK_HIP_ERROR(hA_device.transfer_from(dA));

        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_ger<T, CONJ>(M, N, h_alpha, hx[b], incx, hy[b], incy, hA_cpu[b], lda);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, lda, hA_cpu, hA_host);
            unit_check_general<T>(M, N, batch_count, lda, hA_cpu, hA_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, lda, hA_cpu, hA_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, lda, hA_cpu, hA_device, batch_count);
        }
    }

    if(argus.timing)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasGerBatchedFn(handle,
                                                    M,
                                                    N,
                                                    d_alpha,
                                                    dx.ptr_on_device(),
                                                    incx,
                                                    dy.ptr_on_device(),
                                                    incy,
                                                    dA.ptr_on_device(),
                                                    lda,
                                                    batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_alpha, e_incx, e_incy, e_lda, e_batch_count>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            ger_gflop_count<T>(M, N),
            ger_gbyte_count<T>(M, N),
            hipblas_error_host,
            hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
