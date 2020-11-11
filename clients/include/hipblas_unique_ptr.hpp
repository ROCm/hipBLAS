#ifndef GUARD_HIPBLAS_MANAGE_PTR_HPP
#define GUARD_HIPBLAS_MANAGE_PTR_HPP

#include <memory>

#include "hipblas.h"

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

namespace hipblas
{
    // device_malloc wraps hipMalloc and provides same API as malloc
    static void* device_malloc(size_t byte_size)
    {
        void* pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

} // namespace hipblas

using hipblas_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif
