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
hipblasStatus_t testing_herk_batched(const Arguments& argus)
{
    using U      = real_t<T>;
    bool FORTRAN = argus.fortran;
    auto hipblasHerkBatchedFn
        = FORTRAN ? hipblasHerkBatched<T, U, true> : hipblasHerkBatched<T, U, false>;

    int N           = argus.N;
    int K           = argus.K;
    int lda         = argus.lda;
    int ldc         = argus.ldc;
    int batch_count = argus.batch_count;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    U h_alpha = argus.get_alpha<U>();
    U h_beta  = argus.get_beta<U>();

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K) || batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }

    double             gpu_time_used, hipblas_error_host, hipblas_error_device;
    hipblasLocalHandle handle(argus);

    int    K1     = (transA == HIPBLAS_OP_N ? K : N);
    size_t A_size = size_t(lda) * K1;
    size_t C_size = size_t(ldc) * N;

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_batch_vector<T> hA(A_size, 1, batch_count);
    host_batch_vector<T> hC_host(C_size, 1, batch_count);
    host_batch_vector<T> hC_device(C_size, 1, batch_count);
    host_batch_vector<T> hC_gold(C_size, 1, batch_count);

    device_batch_vector<T> dA(A_size, 1, batch_count);
    device_batch_vector<T> dC(C_size, 1, batch_count);
    device_vector<U>       d_alpha(1);
    device_vector<U>       d_beta(1);

    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());

    hipblas_init(hA, true);
    hipblas_init(hC_host);

    hC_device.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(U), hipMemcpyHostToDevice));

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
    CHECK_HIPBLAS_ERROR(hipblasHerkBatchedFn(handle,
                                             uplo,
                                             transA,
                                             N,
                                             K,
                                             &h_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             &h_beta,
                                             dC.ptr_on_device(),
                                             ldc,
                                             batch_count));

    CHECK_HIP_ERROR(hC_host.transfer_from(dC));

    CHECK_HIP_ERROR(dC.transfer_from(hC_device));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_HIPBLAS_ERROR(hipblasHerkBatchedFn(handle,
                                             uplo,
                                             transA,
                                             N,
                                             K,
                                             d_alpha,
                                             dA.ptr_on_device(),
                                             lda,
                                             d_beta,
                                             dC.ptr_on_device(),
                                             ldc,
                                             batch_count));

    CHECK_HIP_ERROR(hC_device.transfer_from(dC));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        for(int b = 0; b < batch_count; b++)
        {
            cblas_herk<T>(uplo, transA, N, K, h_alpha, hA[b], lda, h_beta, hC_gold[b], ldc);
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(N, N, batch_count, ldc, hC_gold, hC_host);
            unit_check_general<T>(N, N, batch_count, ldc, hC_gold, hC_device);
        }

        if(argus.norm_check)
        {
            hipblas_error_host
                = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_host, batch_count);
            hipblas_error_device
                = norm_check_general<T>('F', N, N, ldc, hC_gold, hC_device, batch_count);
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

            CHECK_HIPBLAS_ERROR(hipblasHerkBatchedFn(handle,
                                                     uplo,
                                                     transA,
                                                     N,
                                                     K,
                                                     d_alpha,
                                                     dA.ptr_on_device(),
                                                     lda,
                                                     d_beta,
                                                     dC.ptr_on_device(),
                                                     ldc,
                                                     batch_count));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used; // in microseconds

        ArgumentModel<e_uplo_option,
                      e_transA_option,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldc,
                      e_batch_count>{}
            .log_args<T>(std::cout,
                         argus,
                         gpu_time_used,
                         herk_gflop_count<T>(N, K),
                         herk_gbyte_count<T>(N, K),
                         hipblas_error_host,
                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
