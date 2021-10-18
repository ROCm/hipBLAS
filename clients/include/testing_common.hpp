/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#pragma once
#ifndef _TESTING_COMMON_HPP_
#define _TESTING_COMMON_HPP_

// do not add special case includes here, keep those in the testing_ file
#include "argument_model.hpp"
#include "bytes.hpp"
#include "cblas_interface.h"
#include "flops.hpp"
#include "hipblas.hpp"
#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif
#include "hipblas_vector.hpp"
#include "near.h"
#include "norm.h"
#include "unit.h"
#include "utility.h"

#endif
