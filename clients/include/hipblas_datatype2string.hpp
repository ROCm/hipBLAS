/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef hipblas_DATATYPE2STRING_H_
#define hipblas_DATATYPE2STRING_H_

#include "hipblas.h"
#include "hipblas.hpp"
#include <ostream>
#include <string>

enum hipblas_initialization
{
    rand_int   = 111,
    trig_float = 222,
    hpl        = 333,
};

inline constexpr auto hipblas_initialization2string(hipblas_initialization init)
{
    switch(init)
    {
    case hipblas_initialization::rand_int:
        return "rand_int";
    case hipblas_initialization::trig_float:
        return "trig_float";
    case hipblas_initialization::hpl:
        return "hpl";
    }
    return "invalid";
}

hipblas_initialization string2hipblas_initialization(const std::string& value);

inline std::ostream& operator<<(std::ostream& os, hipblas_initialization init)
{
    return os << hipblas_initialization2string(init);
}

// Complex output
inline std::ostream& operator<<(std::ostream& os, const hipblasComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipblasDoubleComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

char hipblas2char_operation(hipblasOperation_t value);

char hipblas2char_fill(hipblasFillMode_t value);

char hipblas2char_diagonal(hipblasDiagType_t value);

char hipblas2char_side(hipblasSideMode_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value);

hipblasFillMode_t char2hipblas_fill(char value);

hipblasDiagType_t char2hipblas_diagonal(char value);

hipblasSideMode_t char2hipblas_side(char value);

hipblasDatatype_t string2hipblas_datatype(const std::string& value);

// return precision string for hipblas_datatype
inline constexpr auto hipblas_datatype2string(hipblasDatatype_t type)
{
    switch(type)
    {
    case HIPBLAS_R_16F:
        return "f16_r";
    case HIPBLAS_R_32F:
        return "f32_r";
    case HIPBLAS_R_64F:
        return "f64_r";
    case HIPBLAS_C_16F:
        return "f16_k";
    case HIPBLAS_C_32F:
        return "f32_c";
    case HIPBLAS_C_64F:
        return "f64_c";
    case HIPBLAS_R_8I:
        return "i8_r";
    case HIPBLAS_R_8U:
        return "u8_r";
    case HIPBLAS_R_32I:
        return "i32_r";
    case HIPBLAS_R_32U:
        return "u32_r";
    case HIPBLAS_C_8I:
        return "i8_c";
    case HIPBLAS_C_8U:
        return "u8_c";
    case HIPBLAS_C_32I:
        return "i32_c";
    case HIPBLAS_C_32U:
        return "u32_c";
    case HIPBLAS_R_16B:
        return "bf16_r";
    case HIPBLAS_C_16B:
        return "bf16_c";
    }
    return "invalid";
}

#endif
