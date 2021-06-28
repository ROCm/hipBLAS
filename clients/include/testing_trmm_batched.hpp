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
hipblasStatus_t testing_trmm_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTrmmBatchedFn
        = FORTRAN ? hipblasTrmmBatched<T, true> : hipblasTrmmBatched<T, false>;

    int M   = argus.M;
    int N   = argus.N;
    int lda = argus.lda;
    int ldb = argus.ldb;

    hipblasSideMode_t  side   = char2hipblas_side(argus.side_option);
    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);
    hipblasDiagType_t  diag   = char2hipblas_diagonal(argus.diag_option);
    hipblasStatus_t    status = HIPBLAS_STATUS_SUCCESS;

    int    K      = (side == HIPBLAS_SIDE_LEFT ? M : N);
    size_t A_size = size_t(lda) * K;
    size_t B_size = size_t(ldb) * N;

    int batch_count = argus.batch_count;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(M < 0 || N < 0 || lda < K || ldb < M || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    T h_alpha = argus.get_alpha<T>();

    // host arrays
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hB_host(B_size, 1, batch_count);
    host_batch_vector<T> hB_device(B_size, 1, batch_count);
    host_batch_vector<T> hB_gold(B_size, 1, batch_count);

    // device arrays
    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dB(B_size, 1, batch_count);
    device_vector<T>       d_alpha(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hB_host);
    hB_device.copy_from(hB_host);
    hB_gold.copy_from(hB_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasTrmmBatchedFn(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             M,
                                             N,
                                             &h_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             dB.ptr_on_device(),
                                             ldb,
                                             batch_count));

    CHECK_HIP_ERROR(hB_host.transfer_from(dB));
    CHECK_HIP_ERROR(dB.transfer_from(hB_device));

    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasTrmmBatchedFn(handle,
                                             side,
                                             uplo,
                                             transA,
                                             diag,
                                             M,
                                             N,
                                             d_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             dB.ptr_on_device(),
                                             ldb,
                                             batch_count));

    CHECK_HIP_ERROR(hB_device.transfer_from(dB));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        for(int b = 0; b < batch_count; b++)
        {
            cblas_trmm<T>(side, uplo, transA, diag, M, N, h_alpha, hA[b], lda, hB_gold[b], ldb);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(M, N, batch_count, ldb, hB_gold, hB_host);
            unit_check_general<T>(M, N, batch_count, ldb, hB_gold, hB_device);
        }
        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', M, N, ldb, hB_gold, hB_device, batch_count);
        }
    }

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

            CHECK_HIPBLAS_ERROR(hipblasTrmmBatchedFn(handle,
                                                     side,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     N,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     dB.ptr_on_device(),
                                                     ldb,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_side_option,
                      e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_N,
                      e_lda,
                      e_ldb,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         trmm_gflop_count<T>(M, N, K),
                         trmm_gbyte_count<T>(M, N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
