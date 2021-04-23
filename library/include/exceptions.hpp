/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <exception>

// Convert the current C++ exception to hiblasStatus_t
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent hipblasStatus_t here
inline hipblasStatus_t exception_to_hipblas_status(std::exception_ptr e = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(const hipblasStatus_t& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return HIPBLAS_STATUS_ALLOC_FAILED;
}
catch(...)
{
    return HIPBLAS_STATUS_INTERNAL_ERROR;
}
