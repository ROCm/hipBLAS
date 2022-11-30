#pragma once

#include <stddef.h>
#include <level_zero/ze_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct syclblasHandle *syclblasHandle_t;
hipblasStatus_t syclblasCreate(syclblasHandle_t* handle);
hipblasStatus_t syclblasDestroy(syclblasHandle_t handle);
hipblasStatus_t syclblasSetStream(syclblasHandle_t handle, hipStream_t stream);

void print_me();

#ifdef __cplusplus
}
#endif