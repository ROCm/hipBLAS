/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syrk_ex.hpp"
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

typedef std::tuple<vector<int>, vector<double>, char, char, vector<hipblasDatatype_t>, bool>
    syrk_ex_tuple;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
Advance users only: BrainStorm the parameters but do not make artificial one which invalidates the
matrix.
like lda pairs with M, and "lda must >= M". case "lda < M" will be guarded by argument-checkers
inside API of course.
Yet, the goal of this file is to verify result correctness not argument-checkers.

Representative sampling is sufficient, endless brute-force sampling is not necessary
=================================================================== */

// vector of vector, each vector is a {N, K, lda, ldc};
// add/delete as a group
const vector<vector<int>> matrix_size_range
    = {{-1, -1, -1, -1}, {11, 6, 11, 11}, {16, 15, 16, 16}, {32, 12, 32, 32}, {65, 4, 65, 65}};

// vector, each entry is  {alpha, alphai, beta, betai};
// add/delete single values, like {2.0}
const vector<vector<double>> alpha_beta_range
    = {{-0.5, 1.5, 2.0, 1.5}, {2.0, 1.0, 2.0, 1.0}, {0.0, 0.0, 0.0, 0.0}};

const vector<char> uplo_range = {
    'L',
    'U',
};

const vector<char> transA_range = {'N', 'T'}; //, 'C'}; // conjugate not supported yet.

// a_type, c_type
const vector<vector<hipblasDatatype_t>> precisions{{HIPBLAS_C_8I, HIPBLAS_C_32F},
                                                   {HIPBLAS_C_32F, HIPBLAS_C_32F}};

const bool is_fortran[] = {false, true};

/* ===============Google Unit Test==================================================== */

/* =====================================================================
     BLAS-2 syrk_ex:
=================================================================== */

/* ============================Setup Arguments======================================= */

// Please use "class Arguments" (see utility.hpp) to pass parameters to templated testers;
// Some routines may not touch/use certain "members" of objects "argus".
// like BLAS-1 Scal does not have lda, BLAS-2 GEMV does not have ldb, ldc;
// That is fine. These testers & routines will leave untouched members alone.
// Do not use std::tuple to directly pass parameters to testers
// by std:tuple, you have unpack it with extreme care for each one by like "std::get<0>" which is
// not intuitive and error-prone

Arguments setup_syrk_ex_arguments(syrk_ex_tuple tup)
{

    vector<int>               matrix_size     = std::get<0>(tup);
    vector<double>            alpha_beta      = std::get<1>(tup);
    char                      uplo            = std::get<2>(tup);
    char                      transA          = std::get<3>(tup);
    vector<hipblasDatatype_t> precision_types = std::get<4>(tup);
    bool                      fortran         = std::get<5>(tup);

    Arguments arg;

    // see the comments about matrix_size_range above
    arg.N   = matrix_size[0];
    arg.K   = matrix_size[1];
    arg.lda = matrix_size[2];
    arg.ldc = matrix_size[3];

    arg.alpha  = alpha_beta[0];
    arg.alphai = alpha_beta[1];
    arg.beta   = alpha_beta[2];
    arg.betai  = alpha_beta[3];

    arg.timing = 0;

    arg.a_type = precision_types[0];
    arg.c_type = precision_types[1];

    arg.uplo_option   = uplo;
    arg.transA_option = transA;

    arg.fortran = fortran;

    return arg;
}

class blas2_syrk_ex_gtest : public ::TestWithParam<syrk_ex_tuple>
{
protected:
    blas2_syrk_ex_gtest() {}
    virtual ~blas2_syrk_ex_gtest() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// syrk_ex
TEST_P(blas2_syrk_ex_gtest, syrk_ex_gtest_float)
{
    // GetParam return a tuple. Tee setup routine unpack the tuple
    // and initializes arg(Arguments) which will be passed to testing routine
    // The Arguments data struture have physical meaning associated.
    // while the tuple is non-intuitive.

    Arguments arg = setup_syrk_ex_arguments(GetParam());
    if(arg.transA_option == 'C')
        arg.transA_option = 'T';

    hipblasStatus_t status = testing_syrk_ex(arg);

    // if not success, then the input argument is problematic, so detect the error message
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        if(arg.N < 0 || arg.K < 0 || arg.ldc < arg.N
           || (arg.transA_option == 'N' && arg.lda < arg.N)
           || (arg.transA_option != 'N' && arg.lda < arg.K))
        {
            EXPECT_EQ(HIPBLAS_STATUS_INVALID_VALUE, status);
        }
        else
        {
            // TODO: This is not currently supported in rocBLAS.
            // Also, it appears that cuBLAS documentation is wrong,
            // as we are getting a NOT_SUPPORTED return value with
            // the documented Atype and Ctype.
            EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, status);
        }
    }
}

// notice we are using vector of vector
// so each elment in xxx_range is a avector,
// ValuesIn take each element (a vector) and combine them and feed them to test_p
// The combinations are  { {M, N, lda}, {incx,incy} {alpha} }

INSTANTIATE_TEST_CASE_P(hipblasSyrkEx,
                        blas2_syrk_ex_gtest,
                        Combine(ValuesIn(matrix_size_range),
                                ValuesIn(alpha_beta_range),
                                ValuesIn(uplo_range),
                                ValuesIn(transA_range),
                                ValuesIn(precisions),
                                ValuesIn(is_fortran)));
