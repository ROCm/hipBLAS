/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

using namespace std;

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_rotg_strided_batched(const Arguments& arg)
{
    using U      = real_t<T>;
    bool FORTRAN = arg.fortran;
    auto hipblasRotgStridedBatchedFn
        = FORTRAN ? hipblasRotgStridedBatched<T, U, true> : hipblasRotgStridedBatched<T, U, false>;

    double        stride_scale = arg.stride_scale;
    hipblasStride stride_a     = stride_scale;
    hipblasStride stride_b     = stride_scale;
    hipblasStride stride_c     = stride_scale;
    hipblasStride stride_s     = stride_scale;
    int           batch_count  = arg.batch_count;

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

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

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
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR((hipblasRotgStridedBatchedFn(
            handle, ra, stride_a, rb, stride_b, rc, stride_c, rs, stride_s, batch_count)));

        if(arg.unit_check)
        {
            near_check_general<T>(1, 1, batch_count, 1, stride_a, ca, ra, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_b, cb, rb, rel_error);
            near_check_general<U>(1, 1, batch_count, 1, stride_c, cc, rc, rel_error);
            near_check_general<T>(1, 1, batch_count, 1, stride_s, cs, rs, rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, stride_a, ca, ra, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_b, cb, rb, batch_count);
            hipblas_error_host
                += norm_check_general<U>('F', 1, 1, 1, stride_c, cc, rc, batch_count);
            hipblas_error_host
                += norm_check_general<T>('F', 1, 1, 1, stride_s, cs, rs, batch_count);
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
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR((hipblasRotgStridedBatchedFn(
            handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));

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
        if(arg.norm_check)
        {
            hipblas_error_device
                = norm_check_general<T>('F', 1, 1, 1, stride_a, ca, ra, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_b, cb, rb, batch_count);
            hipblas_error_device
                += norm_check_general<U>('F', 1, 1, 1, stride_c, cc, rc, batch_count);
            hipblas_error_device
                += norm_check_general<T>('F', 1, 1, 1, stride_s, cs, rs, batch_count);
        }
    }

    if(arg.timing)
    {
        device_vector<T> da(size_a);
        device_vector<T> db(size_b);
        device_vector<U> dc(size_c);
        device_vector<T> ds(size_s);
        CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T) * size_a, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U) * size_c, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T) * size_s, hipMemcpyHostToDevice));
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR((hipblasRotgStridedBatchedFn(
                handle, da, stride_a, db, stride_b, dc, stride_c, ds, stride_s, batch_count)));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_batch_count>{}.log_args<T>(std::cout,
                                                   arg,
                                                   gpu_time_used,
                                                   rotg_gflop_count<T, U>(),
                                                   rotg_gbyte_count<T, U>(),
                                                   hipblas_error_host,
                                                   hipblas_error_device);
    }
    return HIPBLAS_STATUS_SUCCESS;
}
