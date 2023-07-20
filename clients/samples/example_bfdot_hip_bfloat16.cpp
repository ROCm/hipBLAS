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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "utility.h"
#include <hipblas/hipblas.h>

/* ============================================================================================ */

// if just using host compiler, hip gives minimum definition of bfloat16, so doing
// most of the work in fp32
void reference_bfdot(int                             n,
                     std::vector<hip_bfloat16>       hx,
                     int                             incx,
                     const std::vector<hip_bfloat16> hy,
                     int                             incy,
                     hip_bfloat16*                   res)
{
    float tmp = 0;
    for(int i = 0; i < n; i++)
        tmp += bfloat16_to_float(hx[i * size_t(incx)]) * bfloat16_to_float(hy[i * size_t(incy)]);
    *res = float_to_bfloat16(tmp);
}

int main()
{
    int             N    = 1024;
    int             incx = 1;
    int             incy = 1;
    hipblasStatus_t status;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<hip_bfloat16> hx(N);
    std::vector<hip_bfloat16> hy(N);
    hip_bfloat16*             dx;
    hip_bfloat16*             dy;
    size_t                    x_size = N * size_t(incx);
    size_t                    y_size = N * size_t(incy);

    double gpu_time_used;

    hipblasHandle_t handle;
    hipblasCreate(&handle);

    // allocate memory on device
    CHECK_HIP_ERROR(hipMalloc(&dx, x_size * sizeof(hip_bfloat16)));
    CHECK_HIP_ERROR(hipMalloc(&dy, y_size * sizeof(hip_bfloat16)));

    // Initial Data on CPU
    // Initializes in range of [1, 3]
    srand(1);
    hipblas_init<hip_bfloat16>(hx, 1, N, incx);
    hipblas_init<hip_bfloat16>(hy, 1, N, incy);

    hip_bfloat16 hipblas_result, gold_result;

    CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), x_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), y_size * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    printf("N        hipblas(us)     \n");

    gpu_time_used = get_time_us(); // in microseconds

    /* =====================================================================
         HIPBLAS C interface
    =================================================================== */

    status = hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST);
    status = hipblasBfdot(handle, N, dx, incx, dy, incy, &hipblas_result);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        CHECK_HIP_ERROR(hipFree(dx));
        CHECK_HIP_ERROR(hipFree(dy));
        hipblasDestroy(handle);
        return status;
    }

    gpu_time_used = get_time_us() - gpu_time_used;

    // verify hipblas_bfdot result
    reference_bfdot(N, hx, incx, hy, incy, &gold_result);
    float gold_resf = bfloat16_to_float(gold_result);
    float hip_resf  = bfloat16_to_float(hipblas_result);
    float diff      = std::abs(gold_resf - hip_resf);

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    if(diff)
    {
        printf("CPU RESULT: %f, GPU_RESULT: %f\n", gold_resf, hip_resf);
        printf("SSCAL TEST FAILS\n");
    }
    else
    {
        printf("SSCAL TEST PASSES\n");
    }

    CHECK_HIP_ERROR(hipFree(dx));
    CHECK_HIP_ERROR(hipFree(dy));
    hipblasDestroy(handle);
    return EXIT_SUCCESS;
}
