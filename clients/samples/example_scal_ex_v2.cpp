/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include <hip/library_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "hipblas.h"
#include "utility.h"

/* ============================================================================================ */

int main()
{
    // Testing scalEx with alpha_type == x_type == f16_r; execution_type = f32_r
    int             N = 10240;
    hipblasStatus_t status;
    hipblasHalf     alpha = float_to_half(10.0f);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<hipblasHalf> hx(N);
    hipblasHalf*             dx;

    double gpu_time_used;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, N * sizeof(hipblasHalf)));

    // Initial Data on CPU
    srand(1);
    hipblas_init<hipblasHalf>(hx, 1, N, 1);

    // copy vector is easy in STL; hz(hx): save a copy in hz which will be output of CPU BLAS
    std::vector<hipblasHalf> hz(hx);

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(hipblasHalf) * N, hipMemcpyHostToDevice));

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    /* =====================================================================
         ROCBLAS  C interface
    =================================================================== */

#ifdef HIPBLAS_V2
    status = hipblasScalEx(handle, N, &alpha, HIP_R_16F, dx, HIP_R_16F, 1, HIP_R_32F);
#else
    status = hipblasScalEx(handle, N, &alpha, HIPBLAS_R_16F, dx, HIPBLAS_R_16F, 1, HIPBLAS_R_32F);
#endif

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("bad status: %d\n", status);
        CHECK_HIP_ERROR(hipFree(dx));
        hipblasDestroy(handle);
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // copy output from device to CPU
    CHECK_HIP_ERROR(hipMemcpy(hx.data(), dx, sizeof(hipblasHalf) * N, hipMemcpyDeviceToHost));

    // verify hipblas_scal result
    bool error_in_element = false;
    for(int i = 0; i < N; i++)
    {
        hipblasHalf cpu_res = float_to_half(half_to_float(hz[i]) * half_to_float(alpha));
        if(cpu_res != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ",
                   i,
                   half_to_float(cpu_res),
                   half_to_float(hx[i]));
            error_in_element = true;
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(error_in_element)
    {
        printf("SCALEX TEST FAILS\n");
    }
    else
    {
        printf("SCALEX TEST PASSES\n");
    }

    CHECK_HIP_ERROR(hipFree(dx));
    hipblasDestroy(handle);
    return EXIT_SUCCESS;
}
