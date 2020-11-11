/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _HIPBLAS_TYPE_DISPATCH_
#define _HIPBLAS_TYPE_DISPATCH_
#include "hipblas.hpp"
#include "utility.h"

// ----------------------------------------------------------------------------
// Calls TEST template based on the argument types. TEST<> is expected to
// return a functor which takes a const Arguments& argument. If the types do
// not match a recognized type combination, then TEST<void> is called.  This
// function returns the same type as TEST<...>{}(arg), usually bool or void.
// ----------------------------------------------------------------------------

// Simple functions which take only one datatype
//
// Even if the function can take mixed datatypes, this function can handle the
// cases where the types are uniform, in which case one template type argument
// is passed to TEST, and the rest are assumed to match the first.
template <template <typename...> class TEST>
auto hipblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
    case HIPBLAS_R_16F:
        return TEST<hipblasHalf>{}(arg);
    case HIPBLAS_R_16B:
        return TEST<hipblasBfloat16>{}(arg);
    case HIPBLAS_R_32F:
        return TEST<float>{}(arg);
    case HIPBLAS_R_64F:
        return TEST<double>{}(arg);
    //  case hipblas_datatype_f16_c:
    //      return TEST<hipblas_half_complex>{}(arg);
    case HIPBLAS_C_32F:
        return TEST<hipblasComplex>{}(arg);
    case HIPBLAS_C_64F:
        return TEST<hipblasDoubleComplex>{}(arg);
    default:
        return TEST<void>{}(arg);
    }
}

// BLAS1 functions
template <template <typename...> class TEST>
auto hipblas_blas1_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, Tb = arg.b_type, To = arg.d_type;
    if(Ti == To)
    {
        if(Tb == Ti)
            return hipblas_simple_dispatch<TEST>(arg);
        else
        { // for csscal and zdscal and complex rotg only
            if(Ti == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F)
                return TEST<hipblasComplex, float>{}(arg);
            else if(Ti == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F)
                return TEST<hipblasDoubleComplex, double>{}(arg);
        }
    }
    else if(Ti == HIPBLAS_C_32F && Tb == HIPBLAS_R_32F)
        return TEST<hipblasComplex, float>{}(arg);
    else if(Ti == HIPBLAS_C_64F && Tb == HIPBLAS_R_64F)
        return TEST<hipblasDoubleComplex, double>{}(arg);
    else if(Ti == HIPBLAS_R_32F && Tb == HIPBLAS_R_32F)
        return TEST<float, float>{}(arg);
    else if(Ti == HIPBLAS_R_64F && Tb == HIPBLAS_R_64F)
        return TEST<double, double>{}(arg);
    //  else if(Ti == hipblas_datatype_f16_c && To == HIPBLAS_R_16F)
    //      return TEST<hipblas_half_complex, hipblasHalf>{}(arg);

    return TEST<void>{}(arg);
}

// BLAS1_ex functions
template <template <typename...> class TEST>
auto hipblas_blas1_ex_dispatch(const Arguments& arg)
{
    // For axpy there are 4 types, alpha_type, x_type, y_type, and execution_type.
    // these currently correspond as follows:
    // alpha_type = arg.a_type
    // x_type     = arg.b_type
    // y_type     = arg.c_type
    // ex_type    = arg.compute_type
    //
    // Currently for axpy we're only supporting a limited number of variants,
    // specifically alpha_type == x_type == y_type, however I'm trying to leave
    // this open to expansion.
    const auto Ta = arg.a_type, Tx = arg.b_type, Ty = arg.c_type, Tex = arg.compute_type;

    if(Ta == Tx && Tx == Ty && Ty == Tex)
    {
        return hipblas_simple_dispatch<TEST>(arg); // Ta == Tx == Ty == Tex
    }
    else if(Ta == Tx && Tx == Ty && Ta == HIPBLAS_R_16F && Tex == HIPBLAS_R_32F)
    {
        return TEST<hipblasHalf, hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if(Ta == Tx && Ta == HIPBLAS_R_16F && Tex == HIPBLAS_R_32F)
    {
        // scal half
        return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
    }
    else if(Ta == HIPBLAS_R_32F && Tx == HIPBLAS_C_32F && Tex == HIPBLAS_C_32F)
    {
        // csscal
        return TEST<float, hipblasComplex, hipblasComplex>{}(arg);
    }
    else if(Ta == HIPBLAS_R_64F && Tx == HIPBLAS_C_64F && Tex == HIPBLAS_C_64F)
    {
        // zdscal
        return TEST<double, hipblasDoubleComplex, hipblasDoubleComplex>{}(arg);
    }

    return TEST<void>{}(arg);
}

// gemm functions
template <template <typename...> class TEST>
auto hipblas_gemm_dispatch(const Arguments& arg)
{
    const auto Ti = arg.a_type, To = arg.c_type, Tc = arg.compute_type;

    if(arg.b_type == Ti && arg.d_type == To)
    {
        if(Ti != To)
        {
            if(Ti == HIPBLAS_R_8I && To == HIPBLAS_R_32I && Tc == To)
                return TEST<int8_t, int32_t, int32_t>{}(arg);
        }
        else if(Tc != To)
        {
            if(To == HIPBLAS_R_16F && Tc == HIPBLAS_R_32F)
            {
                return TEST<hipblasHalf, hipblasHalf, float>{}(arg);
            }
            else if(To == HIPBLAS_R_16B && Tc == HIPBLAS_R_32F)
            {
                return TEST<hipblasBfloat16, hipblasBfloat16, float>{}(arg);
            }
        }
        else
        {
            return hipblas_simple_dispatch<TEST>(arg); // Ti = To = Tc
        }
    }
    return TEST<void>{}(arg);
}

#endif
