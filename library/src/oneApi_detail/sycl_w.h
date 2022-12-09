#pragma once

#include "deps/onemkl.h"
#include <hipblas.h>
#include <level_zero/ze_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclblasHandle* syclblasHandle_t;
hipblasStatus_t syclblas_create(syclblasHandle_t* handle);
hipblasStatus_t syclblas_destroy(syclblasHandle_t handle);
hipblasStatus_t syclblas_set_stream(syclblasHandle_t handle,
                                  unsigned long const* lzHandles,
                                  int                  nHandles,
                                   hipStream_t          stream);
hipblasStatus_t syclblas_get_hipstream(syclblasHandle_t handle, hipStream_t* pStream);
syclQueue_t syclblas_get_sycl_queue(syclblasHandle_t handle);
void syclblas_queue_wait(syclQueue_t queue);

#ifdef __cplusplus
}
#endif
