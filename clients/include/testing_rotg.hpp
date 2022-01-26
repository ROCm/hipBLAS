/* ************************************************************************
 * Copyright 2016-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "testing_common.hpp"

/* ============================================================================================ */

template <typename T>
hipblasStatus_t testing_rotg(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.fortran;
    auto hipblasRotgFn = FORTRAN ? hipblasRotg<T, U, true> : hipblasRotg<T, U, false>;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    host_vector<T> ha(1);
    host_vector<T> hb(1);
    host_vector<U> hc(1);
    host_vector<T> hs(1);

    // Initial data on CPU
    srand(1);
    hipblas_init<T>(ha, 1, 1, 1);
    hipblas_init<T>(hb, 1, 1, 1);
    hipblas_init<U>(hc, 1, 1, 1);
    hipblas_init<T>(hs, 1, 1, 1);

    // CPU BLAS
    host_vector<T> ca = ha;
    host_vector<T> cb = hb;
    host_vector<U> cc = hc;
    host_vector<T> cs = hs;

    // result hipBLAS device
    host_vector<T> ra = ha;
    host_vector<T> rb = hb;
    host_vector<U> rc = hc;
    host_vector<T> rs = hs;

    device_vector<T> da(1);
    device_vector<T> db(1);
    device_vector<U> dc(1);
    device_vector<T> ds(1);

    CHECK_HIP_ERROR(hipMemcpy(da, ha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc, sizeof(U), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(ds, hs, sizeof(T), hipMemcpyHostToDevice));

    if(arg.unit_check || arg.norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR((hipblasRotgFn(handle, ha, hb, hc, hs)));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR((hipblasRotgFn(handle, da, db, dc, ds)));

        CHECK_HIP_ERROR(hipMemcpy(ra, da, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rb, db, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rc, dc, sizeof(U), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(rs, ds, sizeof(T), hipMemcpyDeviceToHost));

        cblas_rotg<T, U>(ca, cb, cc, cs);

        if(arg.unit_check)
        {
            near_check_general(1, 1, 1, ca.data(), ha.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), hb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), hc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), hs.data(), rel_error);

            near_check_general(1, 1, 1, ca.data(), ra.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), rb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), rc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), rs.data(), rel_error);
        }

        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, ca, ha);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cb, hb);
            hipblas_error_host += norm_check_general<U>('F', 1, 1, 1, cc, hc);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cs, hs);

            hipblas_error_device = norm_check_general<T>('F', 1, 1, 1, ca, ra);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cb, rb);
            hipblas_error_device += norm_check_general<U>('F', 1, 1, 1, cc, rc);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cs, rs);
        }
    }

    if(arg.timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

        int runs = arg.cold_iters + arg.iters;
        for(int iter = 0; iter < runs; iter++)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);

            CHECK_HIPBLAS_ERROR((hipblasRotgFn(handle, da, db, dc, ds)));
        }
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<>{}.log_args<T>(std::cout,
                                      arg,
                                      gpu_time_used,
                                      ArgumentLogging::NA_value,
                                      ArgumentLogging::NA_value,
                                      hipblas_error_host,
                                      hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
