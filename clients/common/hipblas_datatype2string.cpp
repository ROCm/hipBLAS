/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "hipblas_datatype2string.hpp"
#include "hipblas.h"

// clang-format off
hipblas_initialization string2hipblas_initialization(const std::string& value)
{
    return
        value == "rand_int"   ? hipblas_initialization::rand_int   :
        value == "trig_float" ? hipblas_initialization::trig_float :
        value == "hpl"        ? hipblas_initialization::hpl        :
        static_cast<hipblas_initialization>(-1);
}
// clang-format on

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

char hipblas2char_operation(hipblasOperation_t value)
{
    switch(value)
    {
    case HIPBLAS_OP_N:
        return 'N';
    case HIPBLAS_OP_T:
        return 'T';
    case HIPBLAS_OP_C:
        return 'C';
    }
    return '\0';
}

char hipblas2char_fill(hipblasFillMode_t value)
{
    switch(value)
    {
    case HIPBLAS_FILL_MODE_UPPER:
        return 'U';
    case HIPBLAS_FILL_MODE_LOWER:
        return 'L';
    case HIPBLAS_FILL_MODE_FULL:
        return 'F';
    }
    return '\0';
}

char hipblas2char_diagonal(hipblasDiagType_t value)
{
    switch(value)
    {
    case HIPBLAS_DIAG_UNIT:
        return 'U';
    case HIPBLAS_DIAG_NON_UNIT:
        return 'N';
    }
    return '\0';
}

char hipblas2char_side(hipblasSideMode_t value)
{
    switch(value)
    {
    case HIPBLAS_SIDE_LEFT:
        return 'L';
    case HIPBLAS_SIDE_RIGHT:
        return 'R';
    case HIPBLAS_SIDE_BOTH:
        return 'B';
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value)
{
    switch(value)
    {
    case 'N':
        return HIPBLAS_OP_N;
    case 'T':
        return HIPBLAS_OP_T;
    case 'C':
        return HIPBLAS_OP_C;
    case 'n':
        return HIPBLAS_OP_N;
    case 't':
        return HIPBLAS_OP_T;
    case 'c':
        return HIPBLAS_OP_C;
    }
    return HIPBLAS_OP_N;
}

hipblasFillMode_t char2hipblas_fill(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'L':
        return HIPBLAS_FILL_MODE_LOWER;
    case 'u':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'l':
        return HIPBLAS_FILL_MODE_LOWER;
    }
    return HIPBLAS_FILL_MODE_LOWER;
}

hipblasDiagType_t char2hipblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_DIAG_UNIT;
    case 'N':
        return HIPBLAS_DIAG_NON_UNIT;
    case 'u':
        return HIPBLAS_DIAG_UNIT;
    case 'n':
        return HIPBLAS_DIAG_NON_UNIT;
    }
    return HIPBLAS_DIAG_NON_UNIT;
}

hipblasSideMode_t char2hipblas_side(char value)
{
    switch(value)
    {
    case 'L':
        return HIPBLAS_SIDE_LEFT;
    case 'R':
        return HIPBLAS_SIDE_RIGHT;
    case 'l':
        return HIPBLAS_SIDE_LEFT;
    case 'r':
        return HIPBLAS_SIDE_RIGHT;
    }
    return HIPBLAS_SIDE_LEFT;
}

// clang-format off
hipblasDatatype_t string2hipblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? HIPBLAS_R_16F  :
        value == "f32_r" || value == "s" ? HIPBLAS_R_32F  :
        value == "f64_r" || value == "d" ? HIPBLAS_R_64F  :
        value == "bf16_r"                ? HIPBLAS_R_16B :
        value == "f16_c"                 ? HIPBLAS_C_16B  :
        value == "f32_c" || value == "c" ? HIPBLAS_C_32F  :
        value == "f64_c" || value == "z" ? HIPBLAS_C_64F  :
        value == "bf16_c"                ? HIPBLAS_C_16B :
        value == "i8_r"                  ? HIPBLAS_R_8I   :
        value == "i32_r"                 ? HIPBLAS_R_32I  :
        value == "i8_c"                  ? HIPBLAS_C_8I   :
        value == "i32_c"                 ? HIPBLAS_C_32I  :
        value == "u8_r"                  ? HIPBLAS_R_8U   :
        value == "u32_r"                 ? HIPBLAS_R_32U  :
        value == "u8_c"                  ? HIPBLAS_C_8U   :
        value == "u32_c"                 ? HIPBLAS_C_32U  :
        static_cast<hipblasDatatype_t>(-1);
}
// clang-format on
