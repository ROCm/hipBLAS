/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_tbsv_batched(const Arguments& argus)
{
    bool FORTRAN = argus.fortran;
    auto hipblasTbsvBatchedFn
        = FORTRAN ? hipblasTbsvBatched<T, true> : hipblasTbsvBatched<T, false>;

    int                M           = argus.M;
    int                K           = argus.K;
    int                incx        = argus.incx;
    int                lda         = argus.lda;
    char               char_uplo   = argus.uplo_option;
    char               char_diag   = argus.diag_option;
    char               char_transA = argus.transA_option;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = argus.batch_count;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(M) * M;
    size_t size_AB  = size_t(lda) * M;
    size_t size_x   = abs_incx * size_t(M);

    hipblasLocalHandle handle(argus);

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    bool invalid_size = M < 0 || K < 0 || lda < K + 1 || !incx || batch_count < 0;
    if(invalid_size || !M || !batch_count)
    {
        hipblasStatus_t actual = hipblasTbsvBatchedFn(
            handle, uplo, transA, diag, M, K, nullptr, lda, nullptr, incx, batch_count);
        EXPECT_HIPBLAS_STATUS(
            actual, (invalid_size ? HIPBLAS_STATUS_INVALID_VALUE : HIPBLAS_STATUS_SUCCESS));
        return actual;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(size_A, 1, batch_count);
    host_batch_vector<T> hAB(size_AB, 1, batch_count);
    host_batch_vector<T> AAT(size_A, 1, batch_count);
    host_batch_vector<T> hb(M, incx, batch_count);
    host_batch_vector<T> hx(M, incx, batch_count);
    host_batch_vector<T> hx_or_b(M, incx, batch_count);

    device_batch_vector<T> dAB(size_AB, 1, batch_count);
    device_batch_vector<T> dx_or_b(M, incx, batch_count);

    CHECK_HIP_ERROR(dAB.memcheck());
    CHECK_HIP_ERROR(dx_or_b.memcheck());

    double gpu_time_used, hipblas_error, cumulative_hipblas_error = 0;

    // Initial Data on CPU
    hipblas_init(hA, true);
    hipblas_init(hx);
    hb.copy_from(hx);
    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        banded_matrix_setup(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], M, M, K);

        prepare_triangular_solve((T*)hA[b], M, (T*)AAT[b], M, char_uplo);
        if(diag == HIPBLAS_DIAG_UNIT)
        {
            make_unit_diagonal(uplo, (T*)hA[b], M, M);
        }

        regular_to_banded(uplo == HIPBLAS_FILL_MODE_UPPER, (T*)hA[b], M, (T*)hAB[b], lda, M, K);

        // Calculate hb = hA*hx;
        cblas_tbmv<T>(uplo, transA, diag, M, K, hAB[b], lda, hb[b], incx);
    }

    hx_or_b.copy_from(hb);

    CHECK_HIP_ERROR(dAB.transfer_from(hAB));
    CHECK_HIP_ERROR(dx_or_b.transfer_from(hx_or_b));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    if(argus.unit_check || argus.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasTbsvBatchedFn(handle,
                                                 uplo,
                                                 transA,
                                                 diag,
                                                 M,
                                                 K,
                                                 dAB.ptr_on_device(),
                                                 lda,
                                                 dx_or_b.ptr_on_device(),
                                                 incx,
                                                 batch_count));

        // copy output from device to CPU
        CHECK_HIP_ERROR(hx_or_b.transfer_from(dx_or_b));

        // Calculating error
        // For norm_check/bench, currently taking the cumulative sum of errors over all batches
        for(int b = 0; b < batch_count; b++)
        {
            hipblas_error = std::abs(vector_norm_1<T>(M, abs_incx, hx[b], hx_or_b[b]));
            if(argus.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * M;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
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

            CHECK_HIPBLAS_ERROR(hipblasTbsvBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     diag,
                                                     M,
                                                     K,
                                                     dAB.ptr_on_device(),
                                                     lda,
                                                     dx_or_b.ptr_on_device(),
                                                     incx,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_diag_option,
                      e_M,
                      e_K,
                      e_lda,
                      e_incx,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         tbsv_gflop_count<T>(M, K),
                         tbsv_gbyte_count<T>(M, K),
                         cumulative_hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
