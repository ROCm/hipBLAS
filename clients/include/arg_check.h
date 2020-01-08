/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef _ARG_CHECK_H
#define _ARG_CHECK_H

#include "hipblas.h"

#ifdef GOOGLE_TEST
#include "gtest/gtest.h"
#endif

void verify_hipblas_status_invalid_value(hipblasStatus_t status, const char* message);

#endif
