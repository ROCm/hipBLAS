/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_set_get_atomics_mode.hpp"
#include "utility.h"
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <vector>

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// only GCC/VS 2010 comes with std::tr1::tuple, but it is unnecessary,  std::tuple is good enough;

typedef std::tuple<bool> set_get_atomics_mode_tuple;

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS set_get_atomics_mode:
=================================================================== */

/* ============================Setup Arguments======================================= */

Arguments setup_set_get_atomics_mode_arguments(set_get_atomics_mode_tuple tup)
{
    Arguments arg;
    arg.fortran = std::get<0>(tup);
    return arg;
}

class set_get_atomics_mode_gtest : public ::TestWithParam<set_get_atomics_mode_tuple>
{
protected:
    set_get_atomics_mode_gtest() {}
    virtual ~set_get_atomics_mode_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_P(set_get_atomics_mode_gtest, default)
{
    Arguments       arg    = setup_set_get_atomics_mode_arguments(GetParam());
    hipblasStatus_t status = testing_set_get_atomics_mode(arg);

    EXPECT_EQ(HIPBLAS_STATUS_SUCCESS, status);
}

INSTANTIATE_TEST_SUITE_P(rocblas_auxiliary_small,
                         set_get_atomics_mode_gtest,
                         Combine(ValuesIn(is_fortran)));
