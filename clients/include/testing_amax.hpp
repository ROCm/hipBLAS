/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "hipblas.hpp"
#include "utility.h"
#include "cblas_interface.h"
#include "norm.h"
#include "unit.h"
#include <complex.h>

using namespace std;

/* ============================================================================================ */

template<typename T>
hipblasStatus_t testing_amax(Arguments argus)
{

    int N = argus.N;
    int incx = argus.incx;

    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    //check to prevent undefined memory allocation error
    if( N < 0 || incx < 0 ){
        status = HIPBLAS_STATUS_INVALID_VALUE;
        return status;
    }

    int sizeX = N * incx;

    //Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<T> hx(sizeX);

    T *dx;
    int *d_rocblas_result;
    int cpu_result, rocblas_result;

    int device_pointer = 1;

    double gpu_time_used, cpu_time_used;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    //allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, sizeX * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&d_rocblas_result, sizeof(int)));

    //Initial Data on CPU
    srand(1);
    hipblas_init<T>(hx, 1, N, incx);

    //copy data from CPU to device, does not work for incx != 1
    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T)*N*incx, hipMemcpyHostToDevice));

    if(argus.timing){
        printf("Idamax: N    rocblas(us)     CPU(us)     error\n");
    }


    /* =====================================================================
         ROCBLAS
    =================================================================== */
    if(argus.timing){
        gpu_time_used = get_time_us();// in microseconds
    }

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
     //hipblasAmax accept both dev/host pointer for the scalar
    if(device_pointer){
        status = hipblasAmax<T>(handle,
                        N,
                        dx, incx,
                        d_rocblas_result);
    }
    else{
        status = hipblasAmax<T>(handle,
                        N,
                        dx, incx,
                        &rocblas_result);
    }

    if (status != HIPBLAS_STATUS_SUCCESS) {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(d_rocblas_result));
        hipblasDestroy(handle);
        return status;
    }

    if(device_pointer)    CHECK_HIP_ERROR(hipMemcpy(&rocblas_result, d_rocblas_result, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.timing){
        gpu_time_used = get_time_us() - gpu_time_used;
    }


    if(argus.unit_check || argus.norm_check){

     /* =====================================================================
                 CPU BLAS
     =================================================================== */
        if(argus.timing){
            cpu_time_used = get_time_us();
        }

        cblas_amax<T>(N,
                    hx.data(), incx,
                    &cpu_result);

        if(argus.timing){
            cpu_time_used = get_time_us() - cpu_time_used;
        }


        if(argus.unit_check){
            unit_check_general<int>(1, 1, 1, &cpu_result, &rocblas_result);
        }

        //if enable norm check, norm check is invasive
        //any typeinfo(T) will not work here, because template deduction is matched in compilation time
        if(argus.norm_check){
            printf("The maximum index cpu=%d, gpu=%d\n", cpu_result, rocblas_result);
        }

    }// end of if unit/norm check


    if(argus.timing){
        printf("    %d    %8.2f    %8.2f     ---     \n", (int)N, gpu_time_used, cpu_time_used);
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(d_rocblas_result));
    hipblasDestroy(handle);
    return HIPBLAS_STATUS_SUCCESS;
}
