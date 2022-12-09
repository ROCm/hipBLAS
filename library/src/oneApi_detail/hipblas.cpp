//#include <hip/hip_runtime.h>
#include "deps/onemkl.h"
#include <algorithm>

#include <functional>
#include <hip/hip_interop.h>
#include <hipblas.h>
#include <exceptions.hpp>
//#include <math.h>

#include "sycl_w.h"

// local functions
static hipblasStatus_t updateSyclHandlesToCrrStream(hipStream_t stream, syclblasHandle_t handle)
{
    // Obtain the handles to the LZ handlers.
    unsigned long lzHandles[4];
    int           nHandles = 4;
    hipGetBackendNativeHandles((uintptr_t)stream, lzHandles, &nHandles);
    //Fix-Me : Should Sycl know hipStream_t??
    syclblas_set_stream(handle, lzHandles, nHandles, stream);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle)
try
{
    // create syclBlas
    syclblas_create((syclblasHandle_t*)handle);

    hipStream_t nullStream = NULL; // default or null stream
    // set stream to default NULL stream
    auto status = updateSyclHandlesToCrrStream(nullStream, (syclblasHandle_t)*handle);
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle)
try
{
    return syclblas_destroy((syclblasHandle_t)handle);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream)
try
{
    return updateSyclHandlesToCrrStream(stream, (syclblasHandle_t)handle);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* pStream)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }
    return syclblas_get_hipstream((syclblasHandle_t)handle, pStream);
}
catch(...)
{
    return exception_to_hipblas_status();
}

// atomics mode - cannot find corresponding atomics mode in oneMKL, default to ALLOWED
hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode)
try
{
    *atomics_mode = HIPBLAS_ATOMICS_ALLOWED;
    return HIPBLAS_STATUS_SUCCESS;
 }
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode)
try
{
    // No op
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasGetInt8Datatype(hipblasHandle_t handle, hipblasInt8Datatype_t * int8Type)
try
{
    *int8Type = HIPBLAS_INT8_DATATYPE_INT8;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasSetInt8Datatype(hipblasHandle_t handle, hipblasInt8Datatype_t int8Type)
try
{
    // No op
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Level-1
// Generic amax which can handle batched/stride/non-batched
hipblasStatus_t ihipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int batchCount, int* results) {
    int64_t *dev_results = nullptr;
    hipError_t hip_status = hipMalloc(&dev_result, sizeof(int64_t)*batchCount);

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    for (int i=0; i<batchCount; ++i) {
        onemklSamax(sycl_queue, n, (x+n*i), incx, (dev_results+i));
    }
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    int64_t* results_host_memory = (int64_t*)malloc(sizeof(int64_t)*batchCount);
    hip_status = hipMemcpy(results_host_memory, dev_result, sizeof(int)*batchCount, hipMemcpyDefault);

    //Fix_Me : Chance of data corruption
    for (auto i=0; i<batchCount; ++i) {
        results[i] = (int)results_host_memory[i];
    }
    hip_status = hipFree(&dev_results);
    free(results_host_memory);

    return HIPBLAS_STATUS_SUCCESS;
}

//amax
hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result)
try
{
    ihipblasIsmax(handle, n, x, incx, 1, result);
/*
    int64_t *dev_result = nullptr;
    hipError_t hip_status = hipMalloc(&dev_result, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklSamax(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    //Fix_Me : Chance of data corruption
    hip_status = hipMemcpy(result, dev_result, sizeof(int), hipMemcpyDefault);
*/
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result)
try
{
    int64_t *dev_result = nullptr;
    hipError_t hip_status = hipMalloc(&dev_result, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklDamax(sycl_queue, n, x, incx, dev_result);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    //Fix_Me : Chance of data corruption
    hip_status = hipMemcpy(result, dev_result, sizeof(int), hipMemcpyDefault);

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
try
{
    int64_t *dev_result = nullptr;
    hipError_t hip_status = hipMalloc(&dev_result, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklCamax(sycl_queue, n, (const float _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    //Fix_Me : Chance of data corruption
    hip_status = hipMemcpy(result, dev_result, sizeof(int), hipMemcpyDefault);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasIzamax(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
try
{
    int64_t *dev_result = nullptr;
    hipError_t hip_status = hipMalloc(&dev_result, sizeof(int64_t));

    auto sycl_queue = syclblas_get_sycl_queue((syclblasHandle_t)handle);
    onemklZamax(sycl_queue, n, (const double _Complex*)x, incx, dev_result);
    syclblas_queue_wait(sycl_queue); // wait until task is completed

    //Fix_Me : Chance of data corruption
    hip_status = hipMemcpy(result, dev_result, sizeof(int), hipMemcpyDefault);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasSscal(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx)
try
{
    onemklSscal(syclblasGetSyclQueue((syclblasHandle_t)handle), n, *alpha, x, incx);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}