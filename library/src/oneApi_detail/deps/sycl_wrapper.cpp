#include <iostream>

#include "sycl.hpp"
#include <ext/oneapi/backend/level_zero.hpp>
#include <include/ze_api.h>
#include <oneapi/mkl.hpp>

#define __HIP_PLATFORM_SPIRV__
#include "hipblas.h"

#include "sycl_w.h"

struct syclblasHandle
{
    syclPlatform_t platform;
    syclDevice_t   device;
    syclContext_t  context;
    syclQueue_t    queue;
    hipStream_t    hip_stream;

    syclblasHandle(void)
        : platform()
        , device()
        , context()
        , queue()
        , hip_stream()
    {
    }

    ~syclblasHandle()
    {
        // Fix_Me: CHIP owns LZ resources hecen need to find better way to release sycl resources here
        //syclQueueDestroy(queue);
        //syclContextDestroy(context);
        //syclDeviceDestroy(device);
        //syclPlatformDestroy(platform);
    }
};

int syclPlatformCreate(syclPlatform_t *obj,
                                  ze_driver_handle_t driver) {
    auto sycl_platform = sycl::ext::oneapi::level_zero::make_platform((pi_native_handle) driver);
    *obj = new syclPlatform_st({sycl_platform});
    return 0;
}

int syclPlatformDestroy(syclPlatform_t obj) {
    delete obj;
    return 0;
}

int syclDeviceCreate(syclDevice_t *obj, syclPlatform_t platform,
                                ze_device_handle_t device) {
    auto sycl_device =
        sycl::ext::oneapi::level_zero::make_device(platform->val, (pi_native_handle) device);
    *obj = new syclDevice_st({sycl_device});
    return 0;
}

int syclDeviceDestroy(syclDevice_t obj) {
    delete obj;
    return 0;
}

int syclContextCreate(syclContext_t *obj, syclDevice_t *devices,
                                 size_t ndevices, ze_context_handle_t context,
                                 int keep_ownership) {
    std::vector<sycl::device> sycl_devices(ndevices);
    for (size_t i = 0; i < ndevices; i++)
        sycl_devices[i] = devices[i]->val;

    auto sycl_context =
        sycl::ext::oneapi::level_zero::make_context(sycl_devices, (pi_native_handle) context, keep_ownership);
    *obj = new syclContext_st({sycl_context});
    return 0;
}

int syclContextDestroy(syclContext_t obj) {
    delete obj;
    return 0;
}

int syclQueueCreate(syclQueue_t *obj, syclContext_t context,
                               ze_command_queue_handle_t queue,
                               int keep_ownership) {
    // XXX: ownership argument only used on master
    auto sycl_queue = sycl::ext::oneapi::level_zero::make_queue(context->val, (pi_native_handle) queue, keep_ownership);
    *obj = new syclQueue_st({sycl_queue});
    return 0;
}

int syclQueueDestroy(syclQueue_t obj) {
    delete obj;
    return 0;
}

int syclEventCreate(syclEvent_t *obj, syclContext_t context,
                               ze_event_handle_t event, int keep_ownership) {
   auto sycl_event = sycl::ext::oneapi::level_zero::make_event(context->val, (pi_native_handle) event, keep_ownership);
   *obj = new syclEvent_st({sycl_event});
   return 0;
}

int syclEventDestroy(syclEvent_t obj) {
   delete obj;
   return 0;
}

hipblasStatus_t syclblas_create(syclblasHandle_t* handle)
{
    if(handle != nullptr)
    {
        *handle = new syclblasHandle();
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t syclblas_destroy(syclblasHandle_t handle)
{
    if(handle != nullptr)
    {
        delete handle;
    }
    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t syclblas_set_stream(syclblasHandle_t     handle,
                                  unsigned long const* lzHandles,
                                  int                  nHandles,
                                  hipStream_t          stream)
{
    if(handle != nullptr)
    {
        handle->hip_stream = stream;
        // Obtain the handles to the LZ constructs.
        assert(nHandles == 4);
        auto hDriver  = (ze_driver_handle_t)lzHandles[0];
        auto hDevice  = (ze_device_handle_t)lzHandles[1];
        auto hContext = (ze_context_handle_t)lzHandles[2];
        auto hQueue   = (ze_command_queue_handle_t)lzHandles[3];

        // Build SYCL platform/device/queue from the LZ handles.
        syclPlatformCreate(&handle->platform, hDriver);
        syclDeviceCreate(&handle->device, handle->platform, hDevice);
        // FIX ME: only 1 device is returned from CHIP-SPV's lzHandles
        syclContextCreate(
            &handle->context, &handle->device, 1 /*ndevices*/, hContext, 1 /*keep_ownership*/);
        syclQueueCreate(&handle->queue, handle->context, hQueue, 1 /* keep ownership */);

        auto asyncExceptionHandler = [](sycl::exception_list exceptions) {
            // Report all asynchronous exceptions that occurred.
            for(std::exception_ptr const& e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch(std::exception& e)
                {
                    std::cerr << "Async exception: " << e.what() << std::endl;
                }
            }

            // Rethrow the first asynchronous exception.
            for(std::exception_ptr const& e : exceptions)
            {
                std::rethrow_exception(e);
            }
        };
    }

    return (handle != nullptr) ? HIPBLAS_STATUS_SUCCESS : HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

hipblasStatus_t syclblas_get_hipstream(syclblasHandle_t handle, hipStream_t* pStream) {
    if (handle == nullptr || pStream == nullptr) {
        return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    *pStream = handle->hip_stream;
    return HIPBLAS_STATUS_SUCCESS;
}
syclQueue_t syclblas_get_sycl_queue(syclblasHandle_t handle)
{
    return handle->queue;
}
void syclblas_queue_wait(syclQueue_t queue) {
    queue->val.wait();
}
