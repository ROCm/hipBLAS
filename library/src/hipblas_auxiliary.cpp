/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hipblas/hipblas.h>

// Convert hipblas_status to string
extern "C" const char* hipblasStatusToString(hipblasStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPBLAS_STATUS_SUCCESS);
        CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
        CASE(HIPBLAS_STATUS_ALLOC_FAILED);
        CASE(HIPBLAS_STATUS_INVALID_VALUE);
        CASE(HIPBLAS_STATUS_MAPPING_ERROR);
        CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
        CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
        CASE(HIPBLAS_STATUS_NOT_SUPPORTED);
        CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
        CASE(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
        CASE(HIPBLAS_STATUS_INVALID_ENUM);
        CASE(HIPBLAS_STATUS_UNKNOWN);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid hipblas_status, we return this string.
    return "<undefined hipblasStatus_t value>";
}
