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
hipblasStatus_t testing_rotg_batched(const Arguments& arg)
{
    bool FORTRAN = arg.fortran;
    auto hipblasRotgBatchedFn
        = FORTRAN ? hipblasRotgBatched<T, U, true> : hipblasRotgBatched<T, U, false>;

    int             batch_count = arg.batch_count;
    hipblasStatus_t status_1    = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2    = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3    = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4    = HIPBLAS_STATUS_SUCCESS;

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

    // Initial Data on CPU
    host_vector<T> ha[batch_count];
    host_vector<T> hb[batch_count];
    host_vector<U> hc[batch_count];
    host_vector<T> hs[batch_count];

    device_batch_vector<T> ba(batch_count, 1);
    device_batch_vector<T> bb(batch_count, 1);

    for(int b = 0; b < batch_count; b++)
    {
        ha[b] = host_vector<T>(1);
        hb[b] = host_vector<T>(1);
        hc[b] = host_vector<U>(1);
        hs[b] = host_vector<T>(1);
    }

    host_vector<T> ca[batch_count];
    host_vector<T> cb[batch_count];
    host_vector<U> cc[batch_count];
    host_vector<T> cs[batch_count];

    srand(1);
    for(int b = 0; b < batch_count; b++)
    {
        hipblas_init<T>(ha[b], 1, 1, 1);
        hipblas_init<T>(hb[b], 1, 1, 1);
        hipblas_init<U>(hc[b], 1, 1, 1);
        hipblas_init<T>(hs[b], 1, 1, 1);
        ca[b] = ha[b];
        cb[b] = hb[b];
        cc[b] = hc[b];
        cs[b] = hs[b];
    }

    for(int b = 0; b < batch_count; b++)
    {
        cblas_rotg<T, U>(ca[b], cb[b], cc[b], cs[b]);
    }

    // Test host
    {
        host_vector<T> ra[batch_count];
        host_vector<T> rb[batch_count];
        host_vector<U> rc[batch_count];
        host_vector<T> rs[batch_count];
        T*             ra_in[batch_count];
        T*             rb_in[batch_count];
        U*             rc_in[batch_count];
        T*             rs_in[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            ra_in[b] = ra[b] = ha[b];
            rb_in[b] = rb[b] = hb[b];
            rc_in[b] = rc[b] = hc[b];
            rs_in[b] = rs[b] = hs[b];
        }

        status_1 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);

        status_2 = (hipblasRotgBatchedFn(handle, ra_in, rb_in, rc_in, rs_in, batch_count));

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
            near_check_general<T>(1, 1, batch_count, 1, ra, ca, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rb, cb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, rc, cc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rs, cs, rel_error);
        }
    }

    // Test device
    {
        device_vector<T*, 0, T> da(batch_count);
        device_vector<T*, 0, T> db(batch_count);
        device_vector<U*, 0, U> dc(batch_count);
        device_vector<T*, 0, T> ds(batch_count);
        device_batch_vector<T>  ba(batch_count, 1);
        device_batch_vector<T>  bb(batch_count, 1);
        device_batch_vector<U>  bc(batch_count, 1);
        device_batch_vector<T>  bs(batch_count, 1);
        for(int b = 0; b < batch_count; b++)
        {
            CHECK_HIP_ERROR(hipMemcpy(ba[b], ha[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bb[b], hb[b], sizeof(T), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bc[b], hc[b], sizeof(U), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(bs[b], hs[b], sizeof(T), hipMemcpyHostToDevice));
        }
        CHECK_HIP_ERROR(hipMemcpy(da, ba, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, bb, sizeof(T*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, bc, sizeof(U*) * batch_count, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, bs, sizeof(T*) * batch_count, hipMemcpyHostToDevice));

        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4 = ((hipblasRotgBatchedFn(handle, da, db, dc, ds, batch_count)));

        if((status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
        {
            hipblasDestroy(handle);
            if(status_3 != HIPBLAS_STATUS_SUCCESS)
                return status_3;
            if(status_4 != HIPBLAS_STATUS_SUCCESS)
                return status_4;
        }

        host_vector<T> ra[batch_count];
        host_vector<T> rb[batch_count];
        host_vector<U> rc[batch_count];
        host_vector<T> rs[batch_count];
        for(int b = 0; b < batch_count; b++)
        {
            ra[b] = host_vector<T>(1);
            rb[b] = host_vector<T>(1);
            rc[b] = host_vector<U>(1);
            rs[b] = host_vector<T>(1);
            CHECK_HIP_ERROR(hipMemcpy(ra[b], ba[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rb[b], bb[b], sizeof(T), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rc[b], bc[b], sizeof(U), hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(rs[b], bs[b], sizeof(T), hipMemcpyDeviceToHost));
        }

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, ra, ca, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rb, cb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, rc, cc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, rs, cs, rel_error);
        }
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
