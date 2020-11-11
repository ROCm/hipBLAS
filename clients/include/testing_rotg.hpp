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
hipblasStatus_t testing_rotg(const Arguments& arg)
{
    bool FORTRAN       = arg.fortran;
    auto hipblasRotgFn = FORTRAN ? hipblasRotg<T, U, true> : hipblasRotg<T, U, false>;

    hipblasStatus_t status_1 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_2 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_3 = HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t status_4 = HIPBLAS_STATUS_SUCCESS;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

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
        status_1          = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
        status_2          = ((hipblasRotgFn(handle, ha, hb, hc, hs)));

        if(arg.unit_check)
        {
            near_check_general(1, 1, 1, ca.data(), ha.data(), rel_error);
            near_check_general(1, 1, 1, cb.data(), hb.data(), rel_error);
            near_check_general(1, 1, 1, cc.data(), hc.data(), rel_error);
            near_check_general(1, 1, 1, cs.data(), hs.data(), rel_error);
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
        status_3 = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE);
        status_4 = ((hipblasRotgFn(handle, da, db, dc, ds)));
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
    }
    if((status_1 != HIPBLAS_STATUS_SUCCESS) || (status_2 != HIPBLAS_STATUS_SUCCESS)
       || (status_3 != HIPBLAS_STATUS_SUCCESS) || (status_4 != HIPBLAS_STATUS_SUCCESS))
    {
        hipblasDestroy(handle);
        if(status_1 != HIPBLAS_STATUS_SUCCESS)
            return status_1;
        if(status_2 != HIPBLAS_STATUS_SUCCESS)
            return status_2;
        if(status_3 != HIPBLAS_STATUS_SUCCESS)
            return status_3;
        if(status_4 != HIPBLAS_STATUS_SUCCESS)
            return status_4;
    }
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
