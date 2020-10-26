/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T, typename U = T>
hipblasStatus_t testing_rotg_strided_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotgStridedBatchedFn
        = FORTRAN ? hipblasRotgStridedBatched<T, U, true> : hipblasRotgStridedBatched<T, U, false>;

    double stride_scale = arg.stride_scale;
    int    stride_a     = stride_scale;
    int    stride_b     = stride_scale;
    int    stride_c     = stride_scale;
    int    stride_s     = stride_scale;
    int    batch_count  = arg.batch_count;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    // check to prevent undefined memory allocation error
    if(batch_count == 0)
    {
        return HIPBLAS_STATUS_SUCCESS;
    }
    else if(batch_count < 0)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    size_t size_a = size_t(stride_a) * size_t(batch_count);
    size_t size_b = size_t(stride_b) * size_t(batch_count);
    size_t size_c = size_t(stride_c) * size_t(batch_count);
    size_t size_s = size_t(stride_s) * size_t(batch_count);

    host_vector<T> ha(size_a);
    host_vector<T> hb(size_b);
    host_vector<U> hc(size_c);
    host_vector<T> hs(size_s);

    // Initial data on CPU
    srand(1);
    hipblas_init<T>(ha, 1, 1, 1, stride_a, batch_count);
    hipblas_init<T>(hb, 1, 1, 1, stride_b, batch_count);
    hipblas_init<U>(hc, 1, 1, 1, stride_c, batch_count);
    hipblas_init<T>(hs, 1, 1, 1, stride_s, batch_count);

    // CPU_BLAS
    host_vector<T> ca = ha;
    host_vector<T> cb = hb;
    host_vector<U> cc = hc;
    host_vector<T> cs = hs;

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rotg<T, U>(ca.data() + b * stride_a,
                         cb.data() + b * stride_b,
                         cc.data() + b * stride_c,
                         cs.data() + b * stride_s);
    }

    // Test host
    {
        host_vector<T> ra = ha;
        host_vector<T> rb = hb;
        host_vector<U> rc = hc;
        host_vector<T> rs = hs;
        status_1          = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        status_2          = ((hipblasRotgStridedBatchedFn(
            handle, ra, stride_a, rb, stride_b, rc, stride_c, rs, stride_s, batch_count)));

        if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_1 != HIPBLAS_STATUS_SUCCESS)
                return status_1;
            if(status_2 != HIPBLAS_STATUS_SUCCESS)
                return status_2;
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
        }
    }

    // Test device
    {
        device_vector<T> da(size_a);
        device_vector<T> db(size_b);
        device_vector<U> dc(size_c);
        device_vector<T> ds(size_s);
        CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U) * size_c, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T) * size_s, hipMemcpyHostToDevice));
        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4 = ((hipblasRotgStridedBatchedFn(
            handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));

        if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_3 != HIPBLAS_STATUS_SUCCESS)
                return status_3;
            if(status_4 != HIPBLAS_STATUS_SUCCESS)
                return status_4;
        }

        host_vector<T> ra(size_a);
        host_vector<T> rb(size_b);
        host_vector<U> rc(size_c);
        host_vector<T> rs(size_s);
        CHECK_HIP_ERROR(hipMemcpy(ra, da, sizeof(T) * size_a, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rb, db, sizeof(T) * size_b, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rc, dc, sizeof(U) * size_c, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rs, ds, sizeof(T) * size_s, hipMemcpyDeviceToHost));

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
