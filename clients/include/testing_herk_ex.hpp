/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "cblas_interface.h"
#include "flops.h"
#include "hipblas.hpp"
#include "hipblas_fortran.hpp"
#include "norm.h"
#include "unit.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

template <typename T, typename SCAL, typename Atype, typename Ctype>
hipblasStatus_t testing_herk_ex_template(hipblasFillMode_t  uplo,
                                         hipblasOperation_t transA,
                                         int                N,
                                         int                K,
                                         SCAL               alpha,
                                         int                lda,
                                         SCAL               beta,
                                         int                ldc,
                                         hipblasDatatype_t  a_type,
                                         hipblasDatatype_t  c_type,
                                         bool               unit_check,
                                         bool               norm_check,
                                         bool               timing,
                                         bool               FORTRAN)
{
    auto hipblasHerkExFn = FORTRAN ? hipblasCherkExFortran : hipblasCherkEx;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    int K1     = (transA == HIPBLAS_OP_N ? K : N);
    int A_size = lda * K1;
    int C_size = ldc * N;

    host_vector<Atype> hA(A_size);
    host_vector<Ctype> hC(C_size);
    host_vector<Ctype> hC2(C_size);
    host_vector<T>     hA_T(C_size);

    device_vector<Atype> dA(A_size);
    device_vector<Ctype> dC(C_size);

    double gpu_time_used, cpu_time_used;
    double hipblasGflops, cblas_gflops, hipblasBandwidth;
    double rocblas_error;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // Initial Data on CPU
    srand(1);
    hipblas_init<Atype>(hA, N, K1, lda);
    hipblas_init<Ctype>(hC, N, N, ldc);

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            hA_T[j + i * ldc].real(float(hA[j + i * ldc].real()));
            hA_T[j + i * ldc].imag(float(hA[j + i * ldc].imag()));
        }
    }

    // copy matrix is easy in STL; hB = hA: save a copy in hB which will be output of CPU BLAS
    // hB = hA;

    // copy data from CPU to device
    hipMemcpy(dA, hA.data(), sizeof(Atype) * A_size, hipMemcpyHostToDevice);
    hipMemcpy(dC, hC.data(), sizeof(Ctype) * C_size, hipMemcpyHostToDevice);

    /* =====================================================================
           ROCBLAS
    =================================================================== */
    if(timing)
    {
        gpu_time_used = get_time_us(); // in microseconds
    }

    for(int iter = 0; iter < 1; iter++)
    {
        status = hipblasHerkExFn(handle,
                                 uplo,
                                 transA,
                                 N,
                                 K,
                                 (SCAL*)&alpha,
                                 dA,
                                 a_type,
                                 lda,
                                 (SCAL*)&beta,
                                 dC,
                                 c_type,
                                 ldc);

        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            hipblasDestroy(handle);
            return status;
        }
    }

    // copy output from device to CPU
    hipMemcpy(hC2.data(), dC, sizeof(Ctype) * C_size, hipMemcpyDeviceToHost);

    if(unit_check)
    {
        /* =====================================================================
           CPU BLAS
        =================================================================== */
        cblas_herk<T>(uplo, transA, N, K, alpha, hA_T.data(), lda, beta, hC.data(), ldc);

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general<Ctype>(N, N, ldc, hC2.data(), hC.data());
    }

    hipblasDestroy(handle);
    return status;
}

hipblasStatus_t testing_herk_ex(Arguments argus)
{
    bool FORTRAN = argus.fortran;

    int N   = argus.N;
    int K   = argus.K;
    int lda = argus.lda;
    int ldc = argus.ldc;

    hipblasFillMode_t  uplo   = char2hipblas_fill(argus.uplo_option);
    hipblasOperation_t transA = char2hipblas_operation(argus.transA_option);

    hipblasDatatype_t a_type = argus.a_type;
    hipblasDatatype_t c_type = argus.c_type;

    float alpha = argus.get_alpha<float>();
    float beta  = argus.get_beta<float>();

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    // argument sanity check, quick return if input parameters are invalid before allocating invalid
    // memory
    if(N < 0 || K < 0 || ldc < N || (transA == HIPBLAS_OP_N && lda < N)
       || (transA != HIPBLAS_OP_N && lda < K))
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(a_type == HIPBLAS_C_8I && c_type == HIPBLAS_C_32F)
    {
        status
            = testing_herk_ex_template<hipblasComplex, float, hipblasInt8Complex, hipblasComplex>(
                uplo,
                transA,
                N,
                K,
                alpha,
                lda,
                beta,
                ldc,
                a_type,
                c_type,
                argus.unit_check,
                argus.norm_check,
                argus.timing,
                FORTRAN);
    }
    else if(a_type == HIPBLAS_C_32F && c_type == HIPBLAS_C_32F)
    {
        status = testing_herk_ex_template<hipblasComplex, float, hipblasComplex, hipblasComplex>(
            uplo,
            transA,
            N,
            K,
            alpha,
            lda,
            beta,
            ldc,
            a_type,
            c_type,
            argus.unit_check,
            argus.norm_check,
            argus.timing,
            FORTRAN);
    }
    else
    {
        status = HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    return status;
}
