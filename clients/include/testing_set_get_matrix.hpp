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
hipblasStatus_t testing_set_get_matrix(const Arguments& argus)
{
    bool FORTRAN            = argus.fortran;
    auto hipblasSetMatrixFn = FORTRAN ? hipblasSetMatrixFortran : hipblasSetMatrix;
    auto hipblasGetMatrixFn = FORTRAN ? hipblasGetMatrixFortran : hipblasGetMatrix;

    int rows = argus.rows;
    int cols = argus.cols;
    int lda  = argus.lda;
    int ldb  = argus.ldb;
    int ldc  = argus.ldc;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(rows < 0 || cols < 0 || lda <= 0 || ldb <= 0 || ldc <= 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Naming: dK is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<T> ha(cols * lda);
    host_vector<T> hb(cols * ldb);
    host_vector<T> hb_ref(cols * ldb);
    host_vector<T> hc(cols * ldc);

    device_vector<T> dc(cols * ldc);

    double             hipblas_error = 0.0, gpu_time_used = 0.0;
    hipblasLocalHandle handle(argus);

    // Initial Data on CPU
    srand(1);
    hipblas_init<T>(ha, rows, cols, lda);
    hipblas_init<T>(hb, rows, cols, ldb);
    hb_ref = hb;
    for(int i = 0; i < cols * ldc; i++)
    {
        hc[i] = 100 + i;
    };
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * ldc * cols, hipMemcpyHostToDevice));
    for(int i = 0; i < cols * ldc; i++)
    {
        hc[i] = 99.0;
    };

    /* =====================================================================
           HIPBLAS
    =================================================================== */
    CHECK_HIPBLAS_ERROR(hipblasSetMatrixFn(rows, cols, sizeof(T), (void*)ha, lda, (void*)dc, ldc));
    CHECK_HIPBLAS_ERROR(hipblasGetMatrixFn(rows, cols, sizeof(T), (void*)dc, ldc, (void*)hb, ldb));

    if(argus.unit_check || argus.norm_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */

        // reference calculation
        for(int i1 = 0; i1 < rows; i1++)
        {
            for(int i2 = 0; i2 < cols; i2++)
            {
                hb_ref[i1 + i2 * ldb] = ha[i1 + i2 * lda];
            }
        }

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        if(argus.unit_check)
        {
            unit_check_general<T>(rows, cols, ldb, hb, hb_ref);
        }
        if(argus.norm_check)
        {
            hipblas_error = norm_check_general<T>('F', rows, cols, ldb, hb, hb_ref);
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

            CHECK_HIPBLAS_ERROR(
                hipblasSetMatrixFn(rows, cols, sizeof(T), (void*)ha, lda, (void*)dc, ldc));
            CHECK_HIPBLAS_ERROR(
                hipblasGetMatrixFn(rows, cols, sizeof(T), (void*)dc, ldc, (void*)hb, ldb));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_M, e_N, e_lda, e_ldb, e_ldc>{}.log_args<T>(
            std::cout,
            argus,
            gpu_time_used,
            ArgumentLogging::NA_value,
            set_get_matrix_gbyte_count<T>(rows, cols),
            hipblas_error);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
