#pragma once

#include "deps/onemkl.h"
#include "deps/sycl.h"
#include <hipblas.h>
#include <level_zero/ze_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclblasHandle* syclblasHandle_t;
hipblasStatus_t                syclblasCreate(syclblasHandle_t* handle);
hipblasStatus_t                syclblasDestroy(syclblasHandle_t handle);
hipblasStatus_t                syclblasSetStream(syclblasHandle_t     handle,
                                                 unsigned long const* lzHandles,
                                                 int                  nHandles,
                                                 hipStream_t          stream);
syclQueue_t                    syclblasGetSyclQueue(syclblasHandle_t handle);

#ifdef __cplusplus
}
#endif
