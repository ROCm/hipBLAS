/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "hipblas.h"
#include "utility.h"

using namespace std;

/* ============================================================================================ */

int main()
{

    int             N      = 10240;
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    float           alpha  = 10.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    vector<float> hx(N);
    vector<float> hz(N);
    float*        dx;

    double gpu_time_used;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    hipMalloc(&dx, N * sizeof(float));

    // Initial Data on CPU
    srand(1);
    hipblas_init<float>(hx, 1, N, 1);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    /* =====================================================================
         ROCBLAS  C interface
    =================================================================== */

    status = hipblasSscal(handle, N, &alpha, dx, 1);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

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
    {
        printf("SSCAL TEST FAILS\n");
    }
    else
    {
        printf("SSCAL TEST PASSES\n");
    }

    hipFree(dx);
    hipblasDestroy(handle);
    return EXIT_SUCCESS;
}
