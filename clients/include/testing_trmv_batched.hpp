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

template <typename T>
hipblasStatus_t testing_trmv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmvBatchedFn
        = FORTRAN ? hipblasTrmvBatched<T, true> : hipblasTrmvBatched<T, false>;

    int M    = argus.M;
    int lda  = argus.lda;
    int incx = argus.incx;

    size_t A_size = size_t(lda) * M;
    size_t X_size = size_t(M) * incx;

    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || lda < M || incx == 0 || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    double             gpu_time_used, hipblas_error;
    hipblasLocalHandle handle(argus);

    // arrays of pointers-to-host on host
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hres(M, incx, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dx(M, incx, batch_count);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dx.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hx);
    hres.copy_from(hx);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dx.transfer_from(hx));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasTrmvBatchedFn(handle,
                                             uplo,
                                             transA,
                                             diag,
                                             M,
                                             dA.ptr_on_device(),
                                             lda,
                                             dx.ptr_on_device(),
                                             incx,
                                             batch_count));

    CHECK_HIP_ERROR(hres.transfer_from(dx));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmv<T>(uplo, transA, diag, M, hA[b], lda, hx[b], incx);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(1, M, batch_count, incx, hx, hres);
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', 1, M, incx, hx, hres, batch_count);
        }
    }

    if(argus.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = argus.cold_iters + argus.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == argus.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR(hipblasTrmvBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dx.ptr_on_device(),
                                                     incx,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_lda,
                      e_incx,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmv_gflop_count<T>(M),
                         trmv_gbyte_count<T>(M),
                         hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
