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
hipblasStatus_t testing_rotg(const Arguments& arg)
{
    using U            = real_t<T>;
    bool FORTRAN       = arg.fortran;
    auto hipblasRotgFn = FORTRAN ? hipblasRotg<T, U, true> : hipblasRotg<T, U, false>;

    double gpu_time_used, hipblas_error_host, hipblas_error_device;

    hipblasLocalHandle handle(arg);

    const U rel_error = std::numeric_limits<U>::epsilon() * 1000;

    host_vector<T> a(1);
    host_vector<T> b(1);
    host_vector<U> c(1);
    host_vector<T> s(1);

    // Initial data on CPU
    srand(1);
    hipblas_init<T>(a, 1, 1, 1);
    hipblas_init<T>(b, 1, 1, 1);
    hipblas_init<U>(c, 1, 1, 1);
    hipblas_init<T>(s, 1, 1, 1);

    // CPU BLAS
    host_vector<T> ca = a;
    host_vector<T> cb = b;
    host_vector<U> cc = c;
    host_vector<T> cs = s;
    cblas_rotg<T, U>(ca, cb, cc, cs);

    // Test host
    {
        host_vector<T> ha = a;
        host_vector<T> hb = b;
        host_vector<U> hc = c;
        host_vector<T> hs = s;
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        CHECK_HIPBLAS_ERROR((hipblasRotgFn(handle, ha, hb, hc, hs)));

        if(arg.unit_check)
        {
            near_check_general(1, 1, 1, ca.data(), ha.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), hb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), hc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), hs.data(), rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error_host = norm_check_general<T>('F', 1, 1, 1, ca, ha);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cb, hb);
            hipblas_error_host += norm_check_general<U>('F', 1, 1, 1, cc, hc);
            hipblas_error_host += norm_check_general<T>('F', 1, 1, 1, cs, hs);
        }
    }

    // Test device
    {
        device_vector<T> da(1);
        device_vector<T> db(1);
        device_vector<U> dc(1);
        device_vector<T> ds(1);
        CHECK_HIP_ERROR(hipMemcpy(da, a, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, b, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, c, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, s, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIPBLAS_ERROR((hipblasRotgFn(handle, da, db, dc, ds)));
        host_vector<T> ha(1);
        host_vector<T> hb(1);
        host_vector<U> hc(1);
        host_vector<T> hs(1);
        CHECK_HIP_ERROR(hipMemcpy(ha, da, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hb, db, sizeof(T), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hc, dc, sizeof(U), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hs, ds, sizeof(T), hipMemcpyDeviceToHost));

        if(arg.unit_check)
        {
            near_check_general(1, 1, 1, ca.data(), ha.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), hb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), hc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), hs.data(), rel_error);
        }
        if(arg.norm_check)
        {
            hipblas_error_device = norm_check_general<T>('F', 1, 1, 1, ca, ha);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cb, hb);
            hipblas_error_device += norm_check_general<U>('F', 1, 1, 1, cc, hc);
            hipblas_error_device += norm_check_general<T>('F', 1, 1, 1, cs, hs);
        }
    }

    if(arg.timing)
    {
        device_vector<T> da(1);
        device_vector<T> db(1);
        device_vector<U> dc(1);
        device_vector<T> ds(1);
        CHECK_HIP_ERROR(hipMemcpy(da, a, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(db, b, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, c, sizeof(U), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, s, sizeof(T), hipMemcpyHostToDevice));
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

        ArgumentModel<e_N>{}.log_args<T>(std::cout,
                                         arg,
                                         gpu_time_used,
                                         rotg_gflop_count<T, U>(),
                                         rotg_gbyte_count<T, U>(),
                                         hipblas_error_host,
                                         hipblas_error_device);
    }

    return HIPBLAS_STATUS_SUCCESS;
}
