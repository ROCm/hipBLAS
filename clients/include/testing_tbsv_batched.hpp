/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

inline void testname_tbsv_batched(const Arguments& arg, std::string& name)
{
    ArgumentModel<e_N, e_incx, e_incy, e_batch_count>{}.test_name(arg, name);
}

template <typename T>
inline hipblasStatus_t testing_tbsv_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasTbsvBatchedFn
        = FORTRAN ? hipblasTbsvBatched<T, true> : hipblasTbsvBatched<T, false>;

    int                M           = arg.M;
    int                K           = arg.K;
    int                incx        = arg.incx;
    int                lda         = arg.lda;
    char               char_uplo   = arg.uplo;
    char               char_diag   = arg.diag;
    char               char_transA = arg.transA;
    hipblasFillMode_t  uplo        = char2hipblas_fill(char_uplo);
    hipblasDiagType_t  diag        = char2hipblas_diagonal(char_diag);
    hipblasOperation_t transA      = char2hipblas_operation(char_transA);
    int                batch_count = arg.batch_count;

    int    abs_incx = incx < 0 ? -incx : incx;
    size_t size_A   = size_t(M) * M;
    size_t size_AB  = size_t(lda) * M;
    size_t size_x   = abs_incx * size_t(M);

    hipblasLocalHandle handle(arg);

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
    hipblas_init_vector(hA, arg, hipblas_client_never_set_nan, true);
    hipblas_init_vector(hx, arg, hipblas_client_alpha_sets_nan, false, true);
    hb.copy_from(hx);

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
    if(arg.unit_check || arg.norm_check)
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
            if(arg.unit_check)
            {
                double tolerance = std::numeric_limits<real_t<T>>::epsilon() * 40 * M;
                unit_check_error(hipblas_error, tolerance);
            }

            cumulative_hipblas_error += hipblas_error;
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
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

        ArgumentModel<e_uplo, e_transA, e_diag, e_M, e_K, e_lda, e_incx, e_batch_count>{}
            .log_args<T>(std::cout,
                         arg,
                         gpu_time_used,
                         tbsv_gflop_count<T>(M, K),
                         tbsv_gbyte_count<T>(M, K),
                         cumulative_hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
