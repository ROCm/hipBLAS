/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified 
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//!  This is the master include file for hipblas, wrapping around rocblas and cublas "version 1"
//
#ifndef HIPBLAS_H
#define HIPBLAS_H
#pragma once

enum hipblasStatus_t {
  HIPBLAS_STATUS_SUCCESS,          // Function succeeds
  HIPBLAS_STATUS_NOT_INITIALIZED,  // HIPBLAS library not initialized
  HIPBLAS_STATUS_ALLOC_FAILED,     // resource allocation failed
  HIPBLAS_STATUS_INVALID_VALUE,    // unsupported numerical value was passed to function
  HIPBLAS_STATUS_MAPPING_ERROR,    // access to GPU memory space failed
  HIPBLAS_STATUS_EXECUTION_FAILED, // GPU program failed to execute
  HIPBLAS_STATUS_INTERNAL_ERROR,    // an internal HIPBLAS operation failed
  HIPBLAS_STATUS_NOT_SUPPORTED     // function not implemented
};
	
enum hipblasOperation_t {
	HIPBLAS_OP_N,
	HIPBLAS_OP_T,
	HIPBLAS_OP_C
};

// Commented out code is from hcBLAS. It provides build time switch
// so rocBLAS interface calls either hcBLAS or NVCC BLAS
// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:
#if defined(__HIP_PLATFORM_HCC__) and not defined (__HIP_PLATFORM_NVCC__)
#include <hcc_detail/hip_blas.h>
#elif defined(__HIP_PLATFORM_NVCC__) and not defined (__HIP_PLATFORM_HCC__)
#include <nvcc_detail/hip_blas.h>
#else 
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif 


	
#endif

