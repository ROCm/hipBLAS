/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipblas.h"
#include "utility.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================================ */

int main()
{
    int             N      = 10240;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    float           alpha  = 10.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    float* hx = (float*)calloc(N, sizeof(*hx));
    float* hz = (float*)calloc(N, sizeof(*hz));
    float* dx;

    double gpu_time_used;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    hipMalloc((void**)&dx, N * sizeof(float));

    // Initial Data on CPU
    srand(1);

    for(int i = 0; i < N; ++i)
        hx[i] = rand();

    memcpy(hz, hx, sizeof(*hz) * N);

    hipMemcpy(dx, hx, sizeof(*dx) * N, hipMemcpyHostToDevice);

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    status = hipblasSscal(handle, N, &alpha, dx, 1);
    if(status != HIPBLAS_STATUS_SUCCESS)
        return status;

    gpu_time_used = get_time_us() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx, dx, sizeof(*hx) * N, hipMemcpyDeviceToHost);

    // verify hipblas_scal result
    bool error_in_element = false;
    for(int i = 0; i < N; i++)
    {
        if(hz[i] * alpha != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
            error_in_element = true;
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(error_in_element)
        printf("SSCAL TEST FAILS\n");
    else
        printf("SSCAL TEST PASSES\n");

    hipFree(dx);
    hipblasDestroy(handle);
    free(hx);
    free(hz);
    return EXIT_SUCCESS;
}
